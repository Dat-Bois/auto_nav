import time
import numpy as np
from auto_nav import Planner, CasSolver, QPSolver


if __name__ == "__main__":

    sample_loop_start_pos = [18, 10, 1.45]
    sample_loop = np.array([
      [20,10,1.45],
      [32,12,1.45],
      [20,16,1.45],
      [14, 14, 1.45],
      [20,10,1.45]
   ])

    drag_start_pos = [3,3,1.45]
    drag_race = np.array([
      [6.9, 4, 1.45],
      # [37.5, 10, 1.45],
      [47.5, 12, 1.45],
      [87.1, 24, 1.45],
      [91, 25, 1.45],
    ])
    gate_height = 1.4
    # course_start_pos = [14, 25, 1.45, 90]
    # course = np.array([
    #     # [8, 22, 1.45, 90],
    #     # [20, 22, gate_height, 90], # extra to ensure point is in front of gate
    #     [30, 19, 1.45, 90],
    #     [46, 22, 1.45, 90],
    #     # [47, 22, gate_height+0.2, 90], # extra to ensure lower height
    #     # [57, 20, 3.9, 90], # extra to ensure height
    #     [63, 20, 4.15, 90], 
    #     [85, 18, 4.15, 90],
    #     [90, 18, 2.90, 180],
    #     [85, 18, 1.45, 270],
    #     [68, 13, 1.45, 270],
    #     [55, 7, 1.45, 270],
    #     [37, 12, 1.45, 270],
    #     [19, 7, 1.45, 270],
    #     [9,14, 1.45, 0],
    #     [8, 22, 1.45, 35],  # Closing the loop back to start
    #     # [14, 25, 1.45, 90]
    # ])

    course_start_pos = [8, 22, gate_height]
    course = np.array([
        [14, 25, gate_height],
        [20, 22, gate_height], # extra to ensure point is in front of gate
        [30, 19, gate_height],
        [46, 22, gate_height],
        [47, 22, gate_height+0.2], # extra to ensure lower height
        [57, 20, 3.9], # extra to ensure height
        [63, 20, 4.15], 
        [85, 18, 4.15],
        [90, 18, 2.90],
        [85, 18, gate_height],
        [68, 13, gate_height],
        [55, 7, gate_height],
        [37, 12, gate_height],
        [19, 7, gate_height],
        [9,14, gate_height],
        [8, 22, gate_height]  # Closing the loop back to start
    ])

    DRONE_START_POS = course_start_pos
    WAYPOINTS = course
    solver = CasSolver()
    planner = Planner(WAYPOINTS, solver)
    planner.set_hard_constraints(max_tolerance=0.1)
    planner.update_state(position=np.array(DRONE_START_POS))
    traj = planner.plan_global(min_height = 1.3)
    profile = solver.profile(traj)
    solver.visualize(traj, WAYPOINTS, profile)
    np.save("course/trajectory_pos_test.npy", traj)
    # time.sleep(1)
    # check loading
    # traj = np.load("temp/trajectory.npy", allow_pickle=True)
    # profile = solver.profile(traj)
    print("Trajectory loaded and visualized successfully!")

    
