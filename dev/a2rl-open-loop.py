import os
import time
import numpy as np
from auto_nav import CasSolver, QPSolver, MAVROS_API, RCLPY_Handler, Euler, Quaternion, Planner

SIM = os.getenv('RUN_SIM', False)

if __name__ == '__main__':
    handler = RCLPY_Handler("mavros_node")
    api = MAVROS_API(handler, sim=SIM)
    api.connect()
    api.set_mode("GUIDED")

    # api.land(at_home=True, blocking=True)
    # api.disconnect()
    # exit(0)

    api.set_gp_origin(-35.3632621, 149.1652374, 10.0)

    api.arm()
    api.takeoff(1, blocking=True)
    
    api.log("Solving trajectory...")
    waypoints = np.array([
      [20,10,1.45],
      [32,12,1.45],
      [20,16,1.45],
      [14, 14, 1.45],
      [20,10,1.45]
   ])

    solver = CasSolver()
    planner = Planner(waypoints, solver)
    planner.set_hard_constraints(max_velocity=2, max_acceleration=3, max_tolerance=0.2)
    new_state = api.get_DroneState()
    planner.update_state(state = new_state)
    traj = planner.plan_global()
    #--------------------------------
    profile = solver.profile(traj)
    # solver.visualize(traj, waypoints, profile)
    if traj is None:
        api.log("Trajectory could not be solved")
        api.land(at_home=SIM, blocking=True)
        api.disconnect()
    
    # traj = solver.temporal_scale(traj)
    api.log("Trajectory solved!")

    # api.log("Setting initial heading...")
    # api.set_heading(traj[0][4], blocking=True)
    # api.log("Executing trajectory...")
    velocities = profile.get_velocity()
    accels = profile.get_acceleration()
    # x y z t yr
    for i, step in enumerate(zip(traj, velocities, accels)):
        # api.set_velocity(step[0], step[1], step[2], step[4])
        '''Ok logically at a timestep what needs to happen:
        1. At a timestep, that is what the pos, vel, accel should be.
        2. But the assumption is you aren't there, you are at the previous timestep. 
        So you give the setpoint of the next timestep, but wait the current timestep.
        '''
        api.set_full_setpoint(vxyz=step[1][:3], axyz=step[2][:3]) #, yaw_rate=step[1][4]
        # api.set_velocity(step[1][0], step[1][1], step[1][2]) #, yaw_rate=step[1][4])
        if i < len(velocities) - 1:
            # sleep = step[1][3] - velocities[i-1][3]
            sleep = velocities[i+1][3] - step[1][3]
        else:
            sleep = 0.1
        starttime = time.time()
        while time.time() - starttime < sleep:
            pt = api.get_local_pose(as_type="point", ground_truth=SIM)
            if pt is not None:
                profile.save_point(np.array([pt.x, pt.y, pt.z]))

    api.log("Finished...")
    api.set_velocity(0, 0, 0, 0)
    # solver.visualize(traj, waypoints, actual_path=profile.get_actual_path())
    # time.sleep(10)
    api.land(at_home=SIM, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")