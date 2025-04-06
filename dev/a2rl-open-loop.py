import os
import time
import numpy as np
from pathlib import Path
from auto_nav import CasSolver, QPSolver, MAVROS_API, RCLPY_Handler, Euler, Quaternion, Planner

SIM = os.getenv('RUN_SIM', False)

if __name__ == '__main__':
    #-------------------------------
    #-- Temp splicing
    traj = np.load("course/trajectory_pos.npy", allow_pickle=True)
    traj_yaw = np.load("course/trajectory_yaw.npy", allow_pickle=True)
    #--

    solver = CasSolver()
    profile = solver.profile(traj)
    profile_yaw = solver.profile(traj_yaw)
    
    handler = RCLPY_Handler("mavros_node")
    api = MAVROS_API(handler, sim=SIM)
    api.connect()
    api.set_mode("GUIDED")

    # api.land(at_home=True, blocking=True)
    # api.disconnect()
    # exit(0)


    if SIM:
        api.set_gp_origin(-35.3632621, 149.1652374, 10.0)
        api.log("Running in simulation mode. No arming required.")
        api.arm()
    else:
        api.set_gp_origin(24.41526617, 54.44013134, 10.0)
        if not api.wait_for_arm():
            api.log("Failed to arm the drone. Exiting...")
            api.disconnect()
            exit(1)

    api.takeoff(1.4, blocking=True)
    # solver.visualize(traj, waypoints, profile)
    if traj is None:
        api.log("Trajectory could not be solved")
        api.land(at_home=SIM, blocking=True)
        api.disconnect()
    
    # traj = solver.temporal_scale(traj)
    api.log("Trajectory solved!")

    api.log("Setting initial heading...")
    api.set_heading(0, blocking=True)
    api.log("Executing trajectory...")
    velocities = profile.get_velocity()
    yaw_vel = profile_yaw.get_velocity()  # Get yaw velocities
    accels = profile.get_acceleration()
    # x y z t yr
    for i, step in enumerate(zip(traj, velocities, accels)):
        # api.set_velocity(step[0], step[1], step[2], step[4])
        '''Ok logically at a timestep what needs to happen:
        1. At a timestep, that is what the pos, vel, accel should be.
        2. But the assumption is you aren't there, you are at the previous timestep. 
        So you give the setpoint of the next timestep, but wait the current timestep.
        '''
        if i < 69: api.set_full_setpoint(pxyz=step[0][:3], vxyz=step[1][:3], axyz=step[2][:3], yaw_rate=0)
        else: 
            api.set_full_setpoint(pxyz=step[0][:3], vxyz=step[1][:3], axyz=step[2][:3], yaw_rate=yaw_vel[i-69][4])
            api.log(f"Setting yaw rate: {yaw_vel[i-69][4]:.2f} at step {i}")
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
    solver.visualize(traj, waypoints, actual_traj=profile.get_actual_path())
    # time.sleep(10)
    api.land(at_home=SIM, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")
