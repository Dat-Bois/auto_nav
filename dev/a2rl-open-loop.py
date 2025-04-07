import os
import time
import numpy as np
from pathlib import Path
from auto_nav import CasSolver, QPSolver, MAVROS_API, RCLPY_Handler, Euler, Quaternion, Planner

from scipy.spatial.transform import Rotation
from cyclone_a2rl.shared.constants import APRIL_GATE_LOCATIONS

SIM = os.getenv('RUN_SIM', False)

def wp_inline_with_gate(gate, distance, invert_yaw = False):
    '''invert_yaw affects only the yaw of the waypoint, not the position'''
    gate_pos = gate[:3]
    gate_yaw = gate[3]
    direction_vector = Rotation.from_euler('z', gate_yaw, degrees=True).apply([1, 0, 0])
    res = np.array([
        *(gate_pos + distance*direction_vector),
        gate_yaw + 180*int(invert_yaw)
    ])
    return res

gates = APRIL_GATE_LOCATIONS
path = [
    wp_inline_with_gate(gates[1], -5), # gate 2
    (*wp_inline_with_gate(gates[1], 2)[:3], gates[2][3]), # gate 2
    wp_inline_with_gate(gates[2], -5), # gate 3
    (*wp_inline_with_gate(gates[2], 2)[:3], gates[3][3]), # gate 3
    wp_inline_with_gate(gates[4], -2), # DG1 top
    wp_inline_with_gate(gates[4], 1), # DG1 top
    wp_inline_with_gate(gates[5], -1), # DG2 top
    wp_inline_with_gate(gates[5], 1, True), # DG2 top
    wp_inline_with_gate(gates[6], 1, True), # DG2 bottom
    wp_inline_with_gate(gates[6], -1, True), # DG2 bottom
    wp_inline_with_gate(gates[7], 1, True), # G4
    wp_inline_with_gate(gates[7], -1, True), # G4
    wp_inline_with_gate(gates[8], 1, True), # G5
    wp_inline_with_gate(gates[8], -1, True), # G5
    wp_inline_with_gate(gates[9], 1, True), # G6
    wp_inline_with_gate(gates[9], -1, True), # G6
    wp_inline_with_gate(gates[10], 1, True), # G7
    wp_inline_with_gate(gates[10], -1, True), # G7
    wp_inline_with_gate(gates[11], -1), # G8
    wp_inline_with_gate(gates[11], 1), # G8
    (8,22,1,35.), # takeoff point
]
path = np.array(path)

if __name__ == '__main__':
    #-------------------------------
    #-- Temp splicing
    traj = np.load("course/trajectory_pos_test.npy", allow_pickle=True)
    traj_yaw = np.load("course/trajectory_yaw.npy", allow_pickle=True)
    waypoints = None

    #temp scale
    traj = CasSolver().temporal_scale(traj, set_time=20)
    traj_yaw = CasSolver().temporal_scale(traj_yaw, set_time=120)
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
        # if not api.wait_for_arm():
        #     api.log("Failed to arm the drone. Exiting...")
        #     api.disconnect()
        #     exit(1)
        api.arm()

    api.takeoff(1.4, blocking=True, timeout=10)
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
    prev_rate = 0
    for i, step in enumerate(zip(traj, velocities, accels)):
        # api.set_velocity(step[0], step[1], step[2], step[4])
        '''Ok logically at a timestep what needs to happen:
        1. At a timestep, that is what the pos, vel, accel should be.
        2. But the assumption is you aren't there, you are at the previous timestep. 
        So you give the setpoint of the next timestep, but wait the current timestep.
        '''
        if i < 69: 
            yaw_rate = 0
        else:
            try:
                yaw_rate = yaw_vel[i-69][4]
            except:
                yaw_rate = prev_rate
        prev_rate = yaw_rate
        # step[1][:3][2] = np.nan # no vertical velocity setpoint
        # step[2][:3][2] = np.nan # no vertical accel setpoint
        api.set_full_setpoint(pxyz=step[0][:3], vxyz=step[1][:3], axyz=step[2][:3], yaw_rate=yaw_rate)
        api.log(f"Time: {time.time()} | Step {i}: pos={step[0][:3]}, vel={step[1][:3]}, accel={step[2][:3]}, yaw_rate={yaw_rate:.2f}")
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
        
        # # Exit early after first gate
        # pt = api.get_local_pose(as_type="point")
        # if pt is not None:
        #     if pt.x >= 18:
        #         api.log("Finished the first gate, exiting early...")
        #         break
    
    for target in path:
        while True:
            pt = api.get_local_pose(as_type="point")
            if pt is not None:
                # Check if the drone is close enough to the target waypoint
                dist = np.linalg.norm(np.array([pt.x, pt.y, pt.z]) - target[:3])
                if dist < 0.15:
                    api.log(f"Reached waypoint {target[:3]}, moving to next...")
                    break
            api.set_full_setpoint(pxyz=target[:3])
            api.log(f"Target: {target}, Current Position: ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f}), Distance to target: {dist:.2f}")
            time.sleep(0.1)

    api.log("Finished...")
    api.set_velocity(0, 0, 0, 0)
    solver.visualize(traj, waypoints, actual_traj=profile.get_actual_path())
    timestmp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    np.save(f"course/actual_trajectory_pos_{timestmp}.npy", profile.get_actual_path(), allow_pickle=True)
    # time.sleep(10)
    api.land(at_home=SIM, blocking=True)
    api.disconnect()
    print("Connection status: ", api.is_connected())
    print("Done!")
