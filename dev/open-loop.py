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

    waypoints = np.array([  [0, 0, 0], 
                           [1, 2, 0],
                           [2, 0, 2], 
                           [3, -2.2, 2], 
                           [1.5, 0, 2], 
                           [5, 1, 1],
                           [6, 0, 0], 
                           [7, 2, 0],
                           [8, 0, 0],
                           [9, 0, 0]
                           ])

    solver = CasSolver()
    solver.set_hard_constraints(velocity_max=2, acceleration_max=1, max_tolerance=0.2)
    position = pose[0].x, pose[0].y, pose[0].z
    traj = solver.solve(position, waypoints, None, pose[2].yaw)
    profile = solver.profile(traj)
    solver.visualize(traj, waypoints, profile)
    if traj is None:
        api.log("Trajectory could not be solved")
        api.land(at_home=True, blocking=True)
        api.disconnect()
    # traj = solver.temporal_scale(traj)
    api.log("Trajectory solved!")
    # X = traj[:, :3].T
    # T = traj[:, 3]
    # velocity = np.gradient(X, T, axis=1)
    # traj = np.hstack((X.T, T.reshape(-1, 1), velocity.T))
    # print(traj.shape)
    # api.log("Setting initial heading...")
    # api.set_heading(traj[0][4], blocking=True)
    api.log("Executing trajectory...")
    # Traj: x y z t y vx vy vz
    for i, step in enumerate(traj):
        api.set_velocity(step[5], step[6], step[7])
        api.set_heading(step[4])
        time.sleep(traj[i+1][3] - step[3] if i < len(traj) - 1 else 0.1)

    api.log("Finished...")
    pose = api.get_local_pose(as_type='point')
    api.set_local_pose(pose.x, pose.y, pose.z)
    time.sleep(10)
    api.land(at_home=True, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")