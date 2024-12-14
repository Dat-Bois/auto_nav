# drone_API.py 
# This file is an interface allowing for control with ArduPilot through ROS2 DDS.
# Written by: Eesh Vij (evij@uci.edu)


import math
import rclpy
import time
import errno
import threading

from geopy import distance
from geopy import point

from .rclpy_handler import RCLPY_Handler, Publisher, Subscriber, Client

# Message types
# Ardupilot messages
from ardupilot_msgs.msg import GlobalPosition
from ardupilot_msgs.srv import ArmMotors, ModeSwitch
# Geographic messages
from geographic_msgs.msg import GeoPoseStamped, GeoPointStamped
# Geometry messages
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
# Sensor messages
from sensor_msgs.msg import Joy, BatteryState, Imu, NavSatFix
# Built-in messages
from builtin_interfaces.msg import Time
# Standard messages
from std_msgs.msg import String, Float32, Int32, Int64
from rosgraph_msgs.msg import Clock

# Mode constants
MODE_STABILIZE = 0
MODE_GUIDED = 4
MODE_LOITER = 5
MODE_RTL = 6
MODE_LAND = 9

# Publisher topics
PUB_GLOBAL_POSITION = Publisher("/ap/cmd_gps_pose", GlobalPosition)
PUB_CMD_VEL = Publisher("/ap/cmd_vel", TwistStamped)
PUB_JOY = Publisher("/ap/joy", Joy)

# Subscriber topics
SUB_BATTERY = Subscriber("/ap/battery/battery0", BatteryState)
SUB_CLOCK = Subscriber("/ap/clock", Clock)
SUB_GEOPOSE = Subscriber("/ap/geopose/filtered", GeoPoseStamped)
SUB_GEPPOINT = Subscriber("/ap/gps_global_origin/filtered", GeoPointStamped)
SUB_IMU = Subscriber("/ap/imu/experimental/data", Imu)
SUB_NAVSAT = Subscriber("/ap/navsat/navsat0", NavSatFix)
SUB_POSE = Subscriber("/ap/pose/filtered", PoseStamped)
SUB_TIME = Subscriber("/ap/time", Time)
SUB_VEL = Subscriber("/ap/twist/filtered", TwistStamped)

# Client topics
CLI_ARM = Client("/ap/arm_motors", ArmMotors)
CLI_MODE = Client("/ap/mode_switch", ModeSwitch)

#---------------------------------#

class drone_API:

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
        return handler.connected

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
        self.handler.edit_topic_subscriber(SUB_BATTERY, self.batt_cb)
        pass

    def batt_cb(self, msg: BatteryState):
        # self.handler.log(f"Battery voltage: {msg.voltage} V")
        pass

    def init_clients(self):
        clients = [v for k, v in globals().items() if isinstance(v, Client)]
        for cli in clients:
            self.handler.create_service_client(cli)

    def pre_check(self, bypass=[False, False, False]):
        '''Performs a pre-check to ensure that the drone is connected, armed, and in guided mode.
        bypass = [connected, armed, mode]'''
        if (not self.is_connected() and not bypass[0]):
            self.handler.log("Not connected to ROS2 DDS. Please connect first...")
            return False
        if (not self.armed and not bypass[1]):
            self.handler.log("Drone is not armed. Please arm the drone first...")
            return False
        if (self.mode != "loiter" and not bypass[2]):
            self.handler.log("Drone is not in loiter mode. Please switch to guided mode first...")
            return False
        return True

    # Data functions

    def get_latlong(self) -> tuple:
        'Returns the latitude and longitude of the drone in degrees. (lat, long)'
        while not self.handler.connected: pass
        data : GeoPoseStamped = SUB_GEOPOSE.get_latest_data()
        return (data.pose.position.latitude, data.pose.position.longitude)

    def get_altitude(self) -> float:
        'Returns the altitude of the drone in meters.'
        while not self.handler.connected: pass
        data : GeoPoseStamped = SUB_GEOPOSE.get_latest_data()
        return data.pose.position.altitude
    
    def quaternion_to_euler(self, x, y, z, w) -> tuple:
        'Converts a quaternion to Euler angles (roll, pitch, yaw)'
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (roll, pitch, yaw)

    def get_orientation(self) -> tuple:
        'Returns the orientation of the drone in Euler angles (roll, pitch, yaw)'
        while not self.handler.connected: pass
        data : GeoPoseStamped = SUB_GEOPOSE.get_latest_data()
        return self.quaternion_to_euler(data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w)

    def get_linear_velocity(self) -> tuple:
        'Returns the velocity of the drone in the x, y, and z directions (m/s).'
        while not self.handler.connected: pass
        data : TwistStamped = SUB_VEL.get_latest_data()
        return (data.twist.linear.x, data.twist.linear.y, data.twist.linear.z)
    
    def get_angular_velocity(self) -> tuple:
        'Returns the angular velocity of the drone in the x, y, and z directions (rad/s).'
        while not self.handler.connected: pass
        data : TwistStamped = SUB_VEL.get_latest_data()
        return (data.twist.angular.x, data.twist.angular.y, data.twist.angular.z)

    # DRONE CONTROL FUNCTIONS

    def arm(self):
        while not self.pre_check(bypass=[False, True, False]): pass
        self.handler.log("Arming motors ...")
        data = ArmMotors.Request()
        data.arm = True
        self.handler.send_service_request(CLI_ARM, data)
        self.handler.log("Motors armed!")
        self.armed = True

    def disarm(self):
        while not self.pre_check(bypass=[False, False, True]): pass
        self.handler.log("Disarming motors ...")
        data = ArmMotors.Request()
        data.arm = False
        self.handler.send_service_request(CLI_ARM, data)
        self.handler.log("Motors disarmed!")
        self.armed = False

    def switch_mode(self, mode : str):
        while not self.handler.connected: pass
        mode_int = globals()["MODE_" + mode.upper()]
        self.handler.log(f"Switching mode to {mode.upper()} ...")
        data = ModeSwitch.Request()
        data.mode = mode_int
        self.handler.send_service_request(CLI_MODE, data)
        self.handler.log(f"Mode switched to {mode.upper()}!")
        self.mode = mode.lower()

    def set_velocity(self, x, y, z):
        'Sets the velocity of the drone in the x, y, and z directions in meters.'
        while not self.pre_check(): pass
        data = TwistStamped()
        data.twist.linear.x = float(x)
        data.twist.linear.y = float(y)
        data.twist.linear.z = float(z)
        self.handler.publish_topic(PUB_CMD_VEL, data)

    def set_joy(self, axes : list):
        'Sets the velocity of the drone using a joystick controller.'
        while not self.pre_check(): pass
        data = Joy()
        data.axes = [float(i) for i in axes]
        self.handler.publish_topic(PUB_JOY, data)

    def takeoff(self, altitude : float):
        'Takes off the drone to the specified altitude in meters.'
        while not self.pre_check(): pass
        self.handler.log(f"Taking off to {altitude} meters ...")
        data = GlobalPosition()
        data.altitude = float(altitude)
        data.coordinate_frame = 6
        # ignore everything besides altitude
        data.type_mask = 1 | 2 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048
        self.handler.publish_topic(PUB_GLOBAL_POSITION, data)
        while self.get_altitude() < altitude-0.1: pass
        self.handler.log("Drone reached target altitude!")

    def land(self):
        'Lands the drone.'
        while not self.pre_check(): pass
        self.handler.log("Landing ...")
        self.switch_mode("land")
        while self.get_altitude() > 0.1: pass
        self.handler.log("Drone landed!")
    
if __name__ == "__main__":
    handler = RCLPY_Handler("drone_API")
    drone = drone_API(handler)
    drone.connect()
    time.sleep(2)
    drone.switch_mode("loiter")
    drone.arm()
    for i in range(10):
        drone.set_joy([1.0, 0, 0, 0])
        time.sleep(0.5)
    time.sleep(5)
    drone.land()
    drone.disconnect()
    print("Connection closed.")