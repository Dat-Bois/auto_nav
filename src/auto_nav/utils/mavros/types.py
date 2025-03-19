from geometry_msgs.msg import Quaternion as ROS_Quaternion
from geometry_msgs.msg import Point as ROS_Point
from geometry_msgs.msg import Twist
import math

class Point:
    def __init__(self, x : float | ROS_Point, y : float = None, z : float = None):
        if y == None:
            self.x = x.x
            self.y = x.y
            self.z = x.z
        else:
            self.x = x
            self.y = y
            self.z = z

    def to_ros(self):
        point = ROS_Point()
        point.x = float(self.x)
        point.y = float(self.y)
        point.z = float(self.z)
        return point
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)


class Euler:
    def __init__(self, roll : float, pitch : float, yaw : float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

class Quaternion:
    def __init__(self, x : float | ROS_Quaternion, y : float = None, z : float = None, w : float = None):
        if y == None:
            self.x = x.x
            self.y = x.y
            self.z = x.z
            self.w = x.w
        else:
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    def to_ros(self):
        quat = ROS_Quaternion()
        quat.x = float(self.x)
        quat.y = float(self.y)
        quat.z = float(self.z)
        quat.w = float(self.w)
        return quat
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z, self.w)
    
    def quaternion_to_euler(self) -> Euler:
        '''
        Converts a quaternion to Euler angles.
        '''
        x, y, z, w = self.to_tuple()
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
    
class DroneState:

    def __init__(self, position : Point, orientation : Quaternion, velocity : Twist):
        self.pos = position
        self.quat = orientation
        self.euler = orientation.quaternion_to_euler()
        self.lin_vel = velocity.linear
        self.ang_vel = velocity.angular