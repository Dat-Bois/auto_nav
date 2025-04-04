import numpy as np
from auto_nav import Planner, CasSolver


if __name__ == "__main__":
    # waypoints = np.array([  [0, 0, 1], 
    #                         [1, 2, 1],
    #                         [2, 0, 3], 
    #                         [3, -2.2, 3], 
    #                         [1.5, 0, 3], 
    #                         [5, 1, 2],
    #                         [6, 0, 1], 
    #                         [7, 2, 1],
    #                         [8, 0, 1],
    #                         [9, 0, 1]
    #                         ])
    waypoints = np.array([
      [20,10,1.45],
      [34,12,1.45],
      [20,16,1.45],
      [12, 14, 1.45],
      [20,10,1.45]
   ])
    solver = CasSolver()
    planner = Planner(waypoints, solver)
    planner.set_hard_constraints(max_velocity=2, max_acceleration=3, max_tolerance=0.2)
    planner.update_state(position=np.array([18, 10, 0]))
    traj = planner.plan_global()
    profile = solver.profile(traj)
    solver.visualize(traj, waypoints, profile)