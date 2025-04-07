from auto_nav import MAVROS_API, RCLPY_Handler
import time
import numpy as np
handler = RCLPY_Handler("mavros_node")
api = MAVROS_API(handler)
api.connect()

while True:

    api.set_full_setpoint(pxyz=[1, 2, 3], vxyz=[4, 5, 6], axyz=[7, 8, 9], yaw_rate=10)
    time.sleep(1)