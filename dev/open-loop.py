import time
import numpy as np
from auto_nav import CasSolver, QPSolver, MAVROS_API, RCLPY_Handler, Euler, Quaternion



if __name__ == '__main__':
    handler = RCLPY_Handler("mavros_node")
    api = MAVROS_API(handler)
    api.connect()
    api.set_mode("GUIDED")

    # api.land(at_home=True, blocking=True)
    # api.disconnect()
    # exit(0)

    api.arm()
    api.set_gimbal(orientation=Euler(0, -10, 0))
    api.takeoff(5, blocking=True)
    
    api.log("Solving trajectory...")
    pose = api.get_local_pose()

    waypoints = np.array([  [0, 0, 1], 
                           [1, 2, 1],
                           [2, 0, 3], 
                           [3, -2.2, 3], 
                           [1.5, 0, 3], 
                           [5, 1, 2],
                           [6, 0, 1], 
                           [7, 2, 1],
                           [8, 0, 1],
                           [9, 0, 1]
                           ])

    solver = CasSolver()
    solver.set_hard_constraints(velocity_max=2, acceleration_max=1, max_tolerance=0.2)
    position = pose[0].x, pose[0].y, pose[0].z
    traj = solver.solve(position, waypoints, None, pose[2].yaw)
    # Manual scale
    max_time = 30
    time_var = traj[:, 3]
    multiplier = max_time / time_var[-1]
    time_var = time_var * multiplier
    traj[:, 3] = time_var
    #--------------------------------
    profile = solver.profile(traj)
    solver.visualize(traj, waypoints, profile)
    if traj is None:
        api.log("Trajectory could not be solved")
        api.land(at_home=True, blocking=True)
        api.disconnect()
    
    # traj = solver.temporal_scale(traj)
    api.log("Trajectory solved!")

    api.log("Setting initial heading...")
    api.set_heading(traj[0][4], blocking=True)
    api.log("Executing trajectory...")
    velocities = profile.get_velocity()
    # x y z t yr
    for i, step in enumerate(velocities):
        api.set_velocity(step[0], step[1], step[2], step[4])
        if i < len(velocities) - 1:
            sleep = velocities[i+1][3] - step[3]
        else:
            sleep = 0.1
        starttime = time.time()
        while time.time() - starttime < sleep:
            pt = api.get_local_pose(as_type="point")
            if pt is not None:
                profile.save_point(np.array([pt.x, pt.y, pt.z]))

    api.log("Finished...")
    api.set_velocity(0, 0, 0, 0)
    solver.visualize(traj, waypoints, actual_path=profile.get_actual_path())
    # time.sleep(10)
    api.land(at_home=True, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")