import time
import numpy as np
from auto_nav import CasSolver, QPSolver, MAVROS_API, RCLPY_Handler, Euler, Quaternion, Planner



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
    planner = Planner(waypoints, solver)
    planner.set_hard_constraints(velocity_max=2, acceleration_max=1, max_tolerance=0.2)
    planner.update_state(state = api.get_DroneState())
    traj = planner.plan_global(set_time=30)
    planner.set_trajectory(traj)
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
    vels = profile.get_velocity()
    # x y z t yr
    vel = planner.next_velocity(velocities = vels)
    while(np.all(vel != np.array([0, 0, 0]))):
        api.set_velocity(vel[0], vel[1], vel[2])
        state = api.get_DroneState()
        planner.update_state(state = state)
        profile.save_point(np.array([state.pos.x, state.pos.y, state.pos.z]))
        time.sleep(0.1)
        vel = planner.next_velocity(velocities = vels)

    api.log("Finished...")
    api.set_velocity(0, 0, 0, 0)
    solver.visualize(traj, waypoints, actual_path=profile.get_actual_path())
    # time.sleep(10)
    api.land(at_home=True, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")