import numpy as np
from auto_nav import Planner, CasSolver, QPSolver


if __name__ == "__main__":
    waypoints = np.array([ 
        [0, 0, 0], 
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
    solver = QPSolver()
    # solver.set_hard_constraints(max_velocity=2, max_acceleration=5, max_tolerance=0.2)
    trajectory = solver.solve(None, waypoints)
    profile = solver.profile(trajectory)
    solver.visualize(trajectory, waypoints, profile)