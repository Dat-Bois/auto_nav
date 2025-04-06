import math
import rclpy
import time
import errno
import numpy as np
from threading import Thread, Lock

from typing import List, Tuple

import rclpy.parameter
from .rclpy_handler import RCLPY_Handler, Publisher, Subscriber, WallTimer, Client
from .types import ROS_Quaternion, Quaternion, Euler, ROS_Point, Point, DroneState
from .sim.gz_truth import GzTopicParser

# MAVROS messages
# Generic services
from mavros_msgs.srv import CommandBool, CommandHome, CommandTOL, SetMode, StreamRate, ParamSet
# Control messages
from mavros_msgs.msg import State, OverrideRCIn, RCIn, Thrust, FullSetpoint
# Waypoint messages
from mavros_msgs.msg import WaypointReached, WaypointList
from mavros_msgs.srv import WaypointSetCurrent, WaypointPull, WaypointPush, WaypointClear

# MAVROS Parameter messages
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# Geographic messages
from geographic_msgs.msg import GeoPoseStamped, GeoPointStamped
# Geometry messages
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Vector3, Vector3Stamped
# Sensor messages
from sensor_msgs.msg import BatteryState, Imu, NavSatFix
# Built-in messages
from builtin_interfaces.msg import Time
# Standard messages
from std_msgs.msg import String, Float32, Float64, Int32, Int64
from rosgraph_msgs.msg import Clock


# Publisher topics
# Control topics
PUB_RCOVERRIDE = Publisher("/mavros/rc/override", OverrideRCIn)
PUB_GLOBAL_SETPOINT = Publisher("/mavros/setpoint_position/global", GeoPoseStamped)
PUB_LOCAL_SETPOINT = Publisher("/mavros/setpoint_position/local", PoseStamped)
PUB_SET_VEL = Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist)
PUB_SET_ANGLE_VEL = Publisher("/mavros/setpoint_attitude/cmd_vel", TwistStamped)
PUB_SET_THRUST = Publisher("/mavros/setpoint_attitude/thrust", Thrust)
PUB_SET_ACCEL = Publisher("/mavros/setpoint_accel/accel", Vector3Stamped)
PUB_SET_FULL = Publisher("/mavros/setpoint_full/cmd_full_setpoint", FullSetpoint)

SIM_SET_GPORIGIN = Publisher("/mavros/global_position/set_gp_origin", GeoPointStamped)

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
SUB_VEL = Subscriber("/mavros/local_position/velocity_local", TwistStamped)
# RC topics
SUB_RC_IN = Subscriber("/mavros/rc/in", RCIn)

# Timers
TIMER_GIMBAL = WallTimer("/mavros/rc/override", 0.1)

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

    def __init__(self, handler: RCLPY_Handler, *, sim : bool = False):
        self.handler = handler
        self.init_topics()
        self.conn_thread = Thread(target=self._connect, daemon=True)
        self.armed = False
        self.mode = None
        
        # When using simulation, set this to True
        self.gz = sim
        self.gimbal_channels = [0,0,0]
        if self.gz:
            self.set_gimbal()
            self.gz_truth = GzTopicParser()

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
        self.init_timers()

    def init_publishers(self):
        publishers = [v for k, v in globals().items() if isinstance(v, Publisher)]
        for pub in publishers:
            self.handler.create_topic_publisher(pub)

    def init_subscribers(self):
        subscribers = [v for k, v in globals().items() if isinstance(v, Subscriber)]
        for sub in subscribers:
            self.handler.create_topic_subscriber(sub)
        self.edit_subscribers()

    def init_timers(self):
        timers = [v for k, v in globals().items() if isinstance(v, WallTimer)]
        for timer in timers:
            self.handler.create_timer(timer)
        self.edit_timers()

    def edit_subscribers(self):
        self.handler.edit_topic_subscriber(SUB_STATE, self.update_state)
        pass

    def edit_timers(self):
        TIMER_GIMBAL.set_func(self._set_gimbal_callback)

    def init_clients(self):
        clients = [v for k, v in globals().items() if isinstance(v, Client)]
        for cli in clients:
            self.handler.create_service_client(cli)

    '''
    ############## - GETTERS - ##############
    '''
    
    def _connected(func):
        def wrapper(self : 'MAVROS_API', *args, **kwargs):
            if self.is_connected():
                return func(self, *args, **kwargs)
            else:
                self.log(f"Not connected to MAVROS! {func}")
                time.sleep(1)
        return wrapper

    @_connected
    def update_state(self, msg : State):
        '''
        Updates the state of the drone.
        '''
        if self.armed == True and msg.armed == False:
            self.log("Drone disarmed externally!")
            self.log('Exiting ...')
            self.disconnect()
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
        data : NavSatFix = SUB_GLOBAL_POSE.get_latest_data(blocking=True)
        alt : Float64 = SUB_REL_ALT.get_latest_data(blocking=True)
        return (data.latitude, data.longitude, data.altitude, alt.data)

    @_connected
    def get_local_pose(self, *, as_type : str = None, ground_truth : bool = False) -> Tuple[Point, Quaternion, Euler] | Point | Quaternion | Euler:
        '''
        Returns three objects: one XYZ, one Quaternion, one Euler.
        ((x, y, z), (x, y, z, w), (roll, pitch, yaw))
        
        Can specify the return type with the as_type parameter.
        "point" -> XYZ
        "quat" -> Quaternion
        "euler" -> Euler angles
        '''
        if ground_truth and self.gz:
            data : PoseStamped = self.gz_truth.get_pose()
        else:
            data : PoseStamped = SUB_POSE.get_latest_data(blocking=False)
        if data == None:
            return None
        point : Point = Point(data.pose.position)
        quat : Quaternion = Quaternion(data.pose.orientation)
        euler = quat.quaternion_to_euler()
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
    def get_local_vel(self) -> Twist:
        '''
        Returns the local velocity of the drone.
        '''
        data : TwistStamped = SUB_VEL.get_latest_data(blocking=False)
        return data.twist
    
    @_connected
    def get_heading(self) -> float:
        '''
        Returns the heading of the drone in degrees.
        '''
        data : Float64 = SUB_HDG.get_latest_data(blocking=True)
        return data.data
    
    @_connected
    def get_rheading(self) -> float:
        '''
        Returns the heading of the drone in radians.
        '''
        return math.radians(self.get_heading())
    
    @_connected
    def get_lheading(self) -> float:
        '''
        Returns the local heading of the drone in degrees.
        '''
        return self.get_local_pose(as_type="euler").yaw
    
    @_connected
    def get_rlheading(self) -> float:
        '''
        Returns the localheading of the drone in radians.
        '''
        return math.radians(self.get_lheading())

    @_connected
    def get_rc_input(self) -> RCIn:
        '''
        Returns the RC input data.
        rssi and channels
        '''
        return SUB_RC_IN.get_latest_data()
    
    @_connected
    def get_DroneState(self):
        '''
        Gets the feedback state of the drone.
        Position, orientation, velocity.
        '''
        pose = self.get_local_pose()
        vel = self.get_local_vel()
        return DroneState(pose[0], pose[1], vel)

    '''
    ############## - SETTERS - ##############
    '''

    '''
    SERVICE BASED FUNCTIONS
    '''

    def _armed_connected(func):
        def wrapper(self : 'MAVROS_API', *args, **kwargs):
            if not self.is_connected():
                self.log("Not connected to MAVROS! Exiting function ...")
                return
            timeout = 10
            while not self.armed:
                self.log('Waiting for drone to be armed ...')
                time.sleep(1)
                timeout -= 1
                if timeout == 0:
                    self.log("Timeout reached. Exiting function ...")
                    return
            return func(self, *args, **kwargs)
        return wrapper

    @_connected
    def arm(self):
        '''
        Arms the drone.
        '''
        data = CommandBool.Request()
        data.value = True
        self.log("Arming motors ...")
        while not self.armed:
            self.handler.send_service_request(CLI_ARM, data)
            time.sleep(1)
        self.log("Motors armed!")

    @_connected
    def wait_for_arm(self, timeout: int = 30):
        '''
        Waits for the drone to be armed.
        Returns True if armed, False if timeout reached.
        '''
        self.log("Waiting for drone to be armed ...")
        start = time.time()
        while not self.armed and (time.time() - start) < timeout:
            time.sleep(0.5)
        if self.armed:
            self.log("Drone is armed!")
            return True
        else:
            self.log("Timeout reached. Drone not armed.")
            return False
            
    @_connected
    def disarm(self):
        '''
        Disarms the drone.
        '''
        data = CommandBool.Request()
        data.value = False
        self.log("Disarming motors ...")
        self.handler.send_service_request(CLI_ARM, data)
        self.log("Motors disarmed!")

    @_connected
    def set_mode(self, mode : str):
        '''
        Sets the mode of the drone.
        '''
        data = SetMode.Request()
        data.custom_mode = mode.upper()
        self.log(f"Switching mode to {mode} ...")
        while self.mode != mode.upper():
            res = self.handler.send_service_request(CLI_SET_MODE, data)
            time.sleep(0.5)
        if res: self.log(f"Mode switched to {mode}!")

    @_armed_connected
    def _takeoff(self, altitude : float):
        '''
        Takes off the drone to the specified altitude.
        '''
        data = CommandTOL.Request()
        data.altitude = float(altitude)
        self.log("Taking off ...")
        self.handler.send_service_request(CLI_TAKEOFF, data)
        self.log("Drone is airborne!")

    @_armed_connected
    def takeoff(self, altitude : float, *, blocking : bool = False, timeout: float = 5):
        '''
        Takes off the drone to the specified altitude.
        '''
        self._takeoff(altitude)
        if blocking:
            start_time = time.time()
            while self.get_local_pose(as_type="point").z < altitude - 0.1: 
                self.log(f"Current altitude: {self.get_local_pose(as_type='point').z:.2f} m | Target altitude: {altitude} m")
                time.sleep(0.1)
                if time.time() - start_time > timeout:
                    self.log(f"Timed out: {timeout} seconds")
                    break
        self.log("Took off")

    @_armed_connected
    def land(self, *, at_home : bool = False, blocking : bool = False):
        '''
        Lands the drone.
        '''
        if at_home:
            self.set_mode("rtl")
        else:
            data = CommandTOL.Request()
            self.log("Landing ...")
            self.handler.send_service_request(CLI_LAND, data)
            self.log("Drone is heading to the ground!")
        if blocking:
            while self.get_local_pose(as_type="point").z > 0.1: pass

    @_connected
    def set_home(self):
        '''
        Sets the home position of the drone to the drones current position.
        '''
        data = CommandHome.Request()
        data.current_gps = True
        self.log("Setting home position ...")
        self.handler.send_service_request(CLI_SET_HOME, data)
        self.log("Home position set!")

    @_connected
    def set_steam_rate(self, stream_id : int, rate : int, on_off : bool):
        '''
        Sets the stream rate of a specific stream.
        '''
        data = StreamRate.Request()
        data.stream_id = stream_id
        data.message_rate = rate
        data.on_off = on_off
        self.log(f"Setting stream rate for stream {stream_id} ...")
        self.handler.send_service_request(CLI_SET_STREAM_RATE, data)
        self.log(f"Stream rate set for stream {stream_id}!")

    @_connected
    def set_param(self, param : str, value : float | int):
        '''
        Sets an Ardupilot parameter on the drone.
        '''
        data = ParamSet.Request()
        data.param_id = param
        if isinstance(value, float):
            data.value.integer = 0
            data.value.real = value
        elif isinstance(value, int):
            data.value.integer = value
            data.value.real = 0
        else:
            self.log(f"Invalid value type {value} for parameter {param}.")
            return
        self.log(f"Setting parameter {param} ...")
        self.handler.send_service_request(CLI_SET_PARAM, data)
        self.log(f"Parameter {param} set!")

    @_connected
    def set_mavros_param(self, topic : str, param : str, value : bool):
        '''
        Sets a ROS parameter on the drone.
        ONLY SUPPORTS BOOL VALUES FOR NOW.
        '''
        if not isinstance(value, bool):
            self.log("Invalid value type. Only bool values are supported for setting ROS params.")
            return
        PARAM_TOPIC = Client(topic, SetParameters)
        self.handler.create_service_client(PARAM_TOPIC)
        data = SetParameters.Request()
        data.parameters = [Parameter(name=param, value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value))]
        self.log(f"Setting ROS parameter {param} ...")
        self.handler.send_service_request(PARAM_TOPIC, data)
        self.log(f"ROS parameter {param} set!")

    '''
    PUBLISHER BASED FUNCTIONS
    '''

    def set_gp_origin(self, lat: float, lon: float, alt: float = 0.0):
        '''
        Sets the global position origin for the simulation.
        This is used to set the GP origin in the simulation.
        '''
        data = GeoPointStamped()
        data.header.stamp = self.handler.get_time()
        data.position.latitude = lat
        data.position.longitude = lon
        data.position.altitude = alt
        self.log(f"Setting GP origin to ({lat}, {lon}, {alt}) ...")
        self.handler.publish_topic(SIM_SET_GPORIGIN, data)

    @_armed_connected
    def set_rc(self, channels : List[int]):
        '''
        Sets the RC channels of the drone.
        There are 18 channels total.
        Can be set between 1200 and 1800.
        To use the actual RC controller, set the channel to 0.
        To ignore changing a channel, set it to 65535.
        '''
        data = OverrideRCIn()
        data.channels = channels
        self.handler.publish_topic(PUB_RCOVERRIDE, data)

    def euler_to_quat(self, euler : Euler) -> Quaternion:
        '''
        Converts Euler angles to a quaternion.
        '''
        roll = math.radians(euler.roll)
        pitch = math.radians(euler.pitch)
        yaw = math.radians(euler.yaw)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return Quaternion(x, y, z, w)

    @_armed_connected
    def set_global_pose(self, lat : float, lon : float, alt : float = None, yaw : float = None):
        '''
        Sets the global pose of the drone.
        Logic not finalized.
        '''
        data = GeoPoseStamped()
        data.header.stamp = self.handler.get_time()
        data.pose.position.latitude = float(lat)
        data.pose.position.longitude = float(lon)
        data.pose.orientation = self.euler_to_quat(Euler(0, 0, -yaw)).to_ros() if yaw != None else self.get_local_pose(as_type="quat").to_ros()
        data.pose.position.altitude = alt if alt != None else self.get_global_pose()[2] #TODO: CONVERT ELLIPSOID TO ASML
        self.handler.publish_topic(PUB_GLOBAL_SETPOINT, data)

    @_armed_connected
    def set_local_pose(self, x : float, y : float, z : float, yaw : float = None):
        '''
        Sets the local pose of the drone.
        WORKS
        '''
        data = PoseStamped()
        data.header.stamp = self.handler.get_time()
        data.pose.position.x = float(x)
        data.pose.position.y = float(y)
        data.pose.position.z = float(z)
        data.pose.orientation = self.euler_to_quat(Euler(0, 0, -yaw)).to_ros() if yaw != None else self.get_local_pose(as_type="quat").to_ros()
        self.handler.publish_topic(PUB_LOCAL_SETPOINT, data)

    @_armed_connected
    def _set_heading(self, yaw : float):
        data = PoseStamped()
        data.header.stamp = self.handler.get_time()
        data.pose.position = self.get_local_pose(as_type="point").to_ros()
        data.pose.orientation = self.euler_to_quat(Euler(0, 0, -yaw)).to_ros()
        self.handler.publish_topic(PUB_LOCAL_SETPOINT, data)

    @_armed_connected
    def set_heading(self, yaw : float, *, blocking : bool = False):
        '''
        Sets the heading of the drone.
        '''
        self._set_heading(yaw)
        rad = math.radians(yaw)
        if blocking:
            while self.get_rlheading() < rad - 0.1 or self.get_rlheading() > rad + 0.1: 
                self.log(f"Current heading: {self.get_rlheading()} | Target heading: {rad}")
                time.sleep(0.1)

    @_armed_connected
    def set_velocity(self, x : float, y : float, z : float, yr : float = None):
        '''
        Sets the linear velocity of the drone.
        WORKS
        '''
        data = Twist()
        data.linear.x = float(x)
        data.linear.y = float(y)
        data.linear.z = float(z)
        if yr != None:
            data.angular.z = float(yr)
        self.handler.publish_topic(PUB_SET_VEL, data)

    @_armed_connected
    def set_angle_velocity(self, roll : float, pitch : float, yaw : float, thrust : float = 0.5):
        '''
        Sets the angular velocity of the drone.
        WORKS
        '''
        data = TwistStamped()
        data2 = Thrust()
        stamp : Time = self.handler.get_time()
        data.header.stamp = stamp
        data2.header.stamp = stamp
        data.twist.angular.x = float(pitch)
        data.twist.angular.y = float(roll)
        data.twist.angular.z = float(yaw)
        data2.thrust = 0.5
        self.handler.publish_topic(PUB_SET_ANGLE_VEL, data)
        self.handler.publish_topic(PUB_SET_THRUST, data2)

    @_armed_connected
    def set_acceleration(self, x : float, y : float, z : float):
        '''
        Sets the acceleration of the drone.
        WORKS
        '''
        data = Vector3Stamped()
        data.header.stamp = self.handler.get_time()
        data.vector.x = float(x)
        data.vector.y = float(y)
        data.vector.z = float(z)
        self.handler.publish_topic(PUB_SET_ACCEL, data)

    def _build_setpoint_typemask(self, pxyz, vxyz, axyz, yaw, yaw_rate):
        # https://mavlink.io/en/messages/common.html#POSITION_TARGET_TYPEMASK
        typemask = 0
        if pxyz is not None:
            if isinstance(pxyz, tuple) or isinstance(pxyz, list):
                pxyz = [np.nan if i is None else i for i in pxyz]
                pxyz = np.array(pxyz)
        else: pxyz = [np.nan, np.nan, np.nan]
        for i in range(3):
            if np.isnan(pxyz[i]): typemask |= (1 << i)

        if vxyz is not None:
            if isinstance(vxyz, tuple) or isinstance(vxyz, list):
                vxyz = [np.nan if i is None else i for i in vxyz]
                vxyz = np.array(vxyz)
        else: vxyz = [np.nan, np.nan, np.nan]
        for i in range(3):
            if np.isnan(vxyz[i]): typemask |= (1 << (i + 3))

        if axyz is not None:
            if isinstance(axyz, tuple) or isinstance(axyz, list):
                axyz = [np.nan if i is None else i for i in axyz]
                axyz = np.array(axyz)
        else: axyz = [np.nan, np.nan, np.nan]
        for i in range(3):
            if np.isnan(axyz[i]): typemask |= (1 << (i + 6))
            
        yaw = float(yaw) if yaw != None else np.nan
        if np.isnan(yaw): typemask |= (1 << 10)

        yaw_rate = np.nan if yaw_rate == None else float(yaw_rate)
        if np.isnan(yaw_rate): yaw_rate = 0.0 #if yawrate is not passed, set it as 0 so its "ignored"

        return typemask, pxyz, vxyz, axyz, yaw, yaw_rate
        
    @_armed_connected
    def set_full_setpoint(self, pxyz : Tuple[float, float, float] | np.ndarray = None, 
                                vxyz : Tuple[float, float, float] | np.ndarray = None, 
                                axyz : Tuple[float, float, float] | np.ndarray = None,
                                yaw: float = None, yaw_rate : float = None, *, typemask : int = None):
        '''
        Sets the full setpoint of the drone.
        NEED TO TEST
        '''
        new_typemask, pxyz, vxyz, axyz, yaw, yaw_rate = self._build_setpoint_typemask(pxyz, vxyz, axyz, yaw, yaw_rate)
        if typemask == None:
            typemask = new_typemask
        if typemask == 3583: # 0b110111111111
            return # all values are None
        data = FullSetpoint()
        data.header.stamp = self.handler.get_time()
        data.type_mask = typemask
        data.position.x = pxyz[0]
        data.position.y = pxyz[1]
        data.position.z = pxyz[2]
        data.velocity.x = vxyz[0]
        data.velocity.y = vxyz[1]
        data.velocity.z = vxyz[2]
        data.acceleration.x = axyz[0]
        data.acceleration.y = axyz[1]
        data.acceleration.z = axyz[2]
        data.yaw = yaw
        data.yaw_rate = yaw_rate
        self.handler.publish_topic(PUB_SET_FULL, data)

    @_connected
    def _set_gimbal_callback(self):
        '''
        Index 5: Roll
        Index 6: Pitch
        Index 7: Yaw
        '''
        if self.gz:
            channels = [65535] * 18
            channels[5] = self.gimbal_channels[0]
            channels[6] = self.gimbal_channels[1]
            channels[7] = self.gimbal_channels[2]
            data = OverrideRCIn()
            data.channels = channels
            self.handler.publish_topic(PUB_RCOVERRIDE, data)

    def set_gimbal(self, *, pitch : float = None, yaw : float = None, roll : float = None, orientation : Quaternion | Euler = None):
        '''
        Sets the pitch and yaw of the gimbal.
        Pitch range: -135 to 45
        Yaw range: -160 to 160
        Roll range: -30 to 30
        WORKS
        '''
        if pitch == None and yaw == None and roll == None and orientation == None:
            pitch, yaw, roll = 0, 0, 0
        if orientation != None:
            if isinstance(orientation, Euler):
                pitch, yaw, roll = orientation.pitch, orientation.yaw, orientation.roll
            elif isinstance(orientation, Quaternion):
                euler = orientation.quaternion_to_euler()
                pitch, yaw, roll = euler.pitch, euler.yaw, euler.roll
            else:
                self.log("Invalid orientation type. Must be Euler or Quaternion.")
                return
        if roll != None:
            self.gimbal_channels[0] = 1100 + int((yaw + 30) / 60 * 800) # roll
        if pitch != None:
            pitch_range = 45 - (-135)
            self.gimbal_channels[1] = 1100 + int((pitch + 135) / pitch_range * 800) # pitch
        if yaw != None:
            self.gimbal_channels[2] = 1100 + int((yaw + 160) / 320 * 800) # yaw

if __name__ == "__main__":
    handler = RCLPY_Handler("mavros_node")
    api = MAVROS_API(handler)
    api.connect()
    api.set_mode("GUIDED")
    api.arm()
    api.set_gimbal(orientation=Euler(0, -10, 0))
    api.takeoff(5)
    while api.get_local_pose(as_type="point").z < 4.9: pass
    api.log("Setting thrust ...")
    pose = api.get_global_pose()
    print(pose)
    for i in range(200):
        # api.set_angle_velocity(0, 0, 1) # 0.2 rad/s
        api.set_velocity(0,2,0,1)
        vel: Twist = api.get_local_vel()
        if(vel!=None):
            print(f"Y Velocity: {vel.linear.y}m/s, Yaw Rate: {vel.angular.z}rad/s")
        time.sleep(0.1)
    api.set_velocity(0,0,0)
    time.sleep(5)
    api.land(at_home=True, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")
