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

    DRONE_START_POS = drag_start_pos
    WAYPOINTS = drag_race
    solver = QPSolver()
    planner = Planner(WAYPOINTS, solver)
    planner.set_hard_constraints(max_tolerance=0.1)
    planner.update_state(position=np.array(DRONE_START_POS))
    traj = planner.plan_global(set_time=10)
    profile = solver.profile(traj)
    solver.visualize(traj, WAYPOINTS, profile)
    np.save("temp/trajectory.npy", traj)
    time.sleep(1)
    # check loading
    traj = np.load("temp/trajectory.npy", allow_pickle=True)
    profile = solver.profile(traj)
    print("Trajectory loaded and visualized successfully!")