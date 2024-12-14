import math
import rclpy
import time
import errno
from threading import Thread, Lock

from rclpy_handler import RCLPY_Handler, Publisher, Subscriber, Client, Euler

# MAVROS messages
# Generic services
from mavros_msgs.srv import CommandBool, CommandHome, CommandTOL, SetMode, StreamRate, ParamSet
# Control messages
from mavros_msgs.msg import State, OverrideRCIn, RCIn, ManualControl, Thrust
# Waypoint messages
from mavros_msgs.msg import WaypointReached, WaypointList
from mavros_msgs.srv import WaypointSetCurrent, WaypointPull, WaypointPush, WaypointClear

# Geographic messages
from geographic_msgs.msg import GeoPoseStamped, GeoPointStamped
# Geometry messages
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Point, Quaternion, Vector3
# Sensor messages
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
# Built-in messages
from builtin_interfaces.msg import Time
# Standard messages
from std_msgs.msg import String, Float32, Float64, Int32, Int64
from rosgraph_msgs.msg import Clock


# Publisher topics
# Control topics
PUB_OVERRIDE_RC = Publisher("/mavros/rc/override", OverrideRCIn)
PUB_GLOBAL_SETPOINT = Publisher("/mavros/setpoint_position/global", GeoPoseStamped)
PUB_LOCAL_SETPOINT = Publisher("/mavros/setpoint_position/local", PoseStamped)
PUB_SET_VEL = Publisher("/mavros/setpoint_attitude/cmd_vel", TwistStamped)
PUB_SET_ATT = Publisher("/mavros/setpoint_attitude/attitude", PoseStamped)
PUB_SET_THRUST = Publisher("/mavros/setpoint_attitude/thrust", Thrust)

# Subscriber topics
# State topics
SUB_STATE = Subscriber("/mavros/state", State)
SUB_BATTERY = Subscriber("/mavros/battery", BatteryState)
# Positional topics
SUB_GLOBAL_POSE = Subscriber("/mavros/global_position/global", NavSatFix)
SUB_REL_ALT = Subscriber("/mavros/global_position/rel_alt", Float64)
SUB_POSE = Subscriber("/mavros/local_position/pose", PoseStamped)
# Orientation topics
SUB_IMU = Subscriber("/mavros/imu/data", Imu)
SUB_HDG = Subscriber("/mavros/global_position/compass_hdg", Float64)
SUB_VEL = Subscriber("/mavros/global_position/gp_vel", TwistStamped)
# RC topics
SUB_RC_IN = Subscriber("/mavros/rc/in", RCIn)

# Client topics
CLI_ARM = Client("/mavros/cmd/arming", CommandBool)
CLI_SET_HOME = Client("/mavros/cmd/set_home", CommandHome)
CLI_TAKEOFF = Client("/mavros/cmd/takeoff", CommandTOL)
CLI_LAND = Client("/mavros/cmd/land", CommandTOL)
CLI_SET_MODE = Client("/mavros/set_mode", SetMode)
CLI_SET_STREAM_RATE = Client("/mavros/set_stream_rate", StreamRate)
CLI_SET_PARAM = Client("/mavros/param/set", ParamSet)

#---------------------------------#
class MAVROS_API:

    def __init__(self, handler: RCLPY_Handler):
        self.handler = handler
        self.init_topics()
        self.conn_thread = Thread(target=self._connect, daemon=True)
        self.armed = False
        self.mode = "loiter"

    def connect(self):
        self.handler.log("Starting connection thread ...")
        self.conn_thread.start()

    def _connect(self):
        self.handler.connect()

    def disconnect(self):
        return self.handler.disconnect()
    
    def is_connected(self):
        return self.handler.connected

    def log(self, msg : str):
        self.handler.log(msg)

    def init_topics(self):
        self.init_publishers()
        self.init_subscribers()
        self.init_clients()

    def init_publishers(self):
        publishers = [v for k, v in globals().items() if isinstance(v, Publisher)]
        for pub in publishers:
            self.handler.create_topic_publisher(pub)

    def init_subscribers(self):
        subscribers = [v for k, v in globals().items() if isinstance(v, Subscriber)]
        for sub in subscribers:
            self.handler.create_topic_subscriber(sub)
        self.edit_subscribers()

    def edit_subscribers(self):
        self.handler.edit_topic_subscriber(SUB_STATE, self.update_state)
        pass

    def init_clients(self):
        clients = [v for k, v in globals().items() if isinstance(v, Client)]
        for cli in clients:
            self.handler.create_service_client(cli)

    # GETTERS
    
    def _connected(func):
        def wrapper(self : 'MAVROS_API', *args, **kwargs):
            if self.is_connected():
                return func(self, *args, **kwargs)
            else:
                self.log("Not connected to MAVROS!")
        return wrapper

    @_connected
    def update_state(self, msg : State):
        '''
        Updates the state of the drone.
        '''
        self.armed = msg.armed
        self.mode = msg.mode
    
    @_connected
    def get_battery(self) -> BatteryState:
        '''
        Returns the battery state of the drone.
        '''
        return SUB_BATTERY.get_latest_data(blocking=True)
    
    @_connected
    def get_global_pose(self) -> NavSatFix:
        '''
        Returns a tuple in LLAA format.
        (lat, lon, ellipsoid_alt, relative_alt)
        '''
        data : NavSatFix = SUB_GLOBAL_POSE.get_latest_data()
        alt : Float64 = SUB_REL_ALT.get_latest_data()
        return (data.latitude, data.longitude, data.altitude, alt.data)

    def quaternion_to_euler(self, x, y, z, w):
        '''
        Converts a quaternion to Euler angles.
        '''
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.degrees(math.atan2(t3, t4))

        return Euler(roll_x, pitch_y, yaw_z)

    @_connected
    def get_local_pose(self, *, as_type : str = None) -> PoseStamped:
        '''
        Returns three objects: one XYZ, one Quaternion, one Euler.
        ((x, y, z), (x, y, z, w), (roll, pitch, yaw))
        
        Can specify the return type with the as_type parameter.
        "point" -> XYZ
        "quat" -> Quaternion
        "euler" -> Euler angles
        '''
        data : PoseStamped = SUB_POSE.get_latest_data()
        point : Point = data.pose.position
        quat : Quaternion = data.pose.orientation
        euler = self.quaternion_to_euler(quat.x, quat.y, quat.z, quat.w)
        if as_type == "point":
            return point
        elif as_type == "quat":
            return quat
        elif as_type == "euler":
            return euler
        elif as_type == None:
            return (point, quat, euler)
        else:
            self.log("Invalid as_type, returning all data ...")
        return (point, quat, euler)
    
    @_connected
    def get_imu_vel(self) -> Imu:
        '''
        Returns the linear and angular velocity of the drone.
        '''
        return SUB_IMU.get_latest_data()
    
    @_connected
    def get_global_vel(self) -> Twist:
        '''
        Returns the global velocity of the drone.
        '''
        data : TwistStamped = SUB_VEL.get_latest_data()
        return data.twist
    
    @_connected
    def get_heading(self) -> float:
        '''
        Returns the heading of the drone in degrees.
        '''
        data : Float64 = SUB_HDG.get_latest_data()
        return data.data
    
    @_connected
    def get_rc_input(self) -> RCIn:
        '''
        Returns the RC input data.
        rssi and channels
        '''
        return SUB_RC_IN.get_latest_data()


if __name__ == "__main__":
    handler = RCLPY_Handler("mavros_node")
    api = MAVROS_API(handler)
    api.connect()
    for i in range(10):
        print(api.get_battery())
        time.sleep(0.5)
    time.sleep(5)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")