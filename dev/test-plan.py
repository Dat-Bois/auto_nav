import time
import numpy as np
from auto_nav import Planner, CasSolver, QPSolver


if __name__ == "__main__":

    DRONE_START_POS = [18, 10, 1.2]
    waypoints = np.array([
      [20,10,1.45],
      [32,12,1.45],
      [20,16,1.45],
      [14, 14, 1.45],
      [20,10,1.45]
   ])
    solver = CasSolver()
    planner = Planner(waypoints, solver)
    planner.set_hard_constraints(max_tolerance=0.2)
    planner.update_state(position=np.array(DRONE_START_POS))
    traj = planner.plan_global(set_time=15)
    profile = solver.profile(traj)
    solver.visualize(traj, waypoints, profile)
    np.save("temp/trajectory.npy", traj)
    time.sleep(1)
    # check loading
    traj = np.load("temp/trajectory.npy", allow_pickle=True)
    profile = solver.profile(traj)
    solver.visualize(traj, waypoints, profile)
    print("Trajectory loaded and visualized successfully!")