import math
import rclpy
import time
import errno
import threading

from rclpy_handler import RCLPY_Handler, Publisher, Subscriber, Client

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
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
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
        self.conn_thread = threading.Thread(target=self._connect, daemon=True)
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
        # self.handler.edit_topic_subscriber(SUB_BATTERY, self.batt_cb)
        pass

    def init_clients(self):
        clients = [v for k, v in globals().items() if isinstance(v, Client)]
        for cli in clients:
            self.handler.create_service_client(cli)


if __name__ == "__main__":
    handler = RCLPY_Handler("mavros_node")
    api = MAVROS_API(handler)
    api.connect()
    time.sleep(5)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")