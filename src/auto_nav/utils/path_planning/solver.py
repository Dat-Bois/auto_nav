import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from typing import List, Tuple

'''
This point of this file is to provide a solver for the path planning problem.
We will first define a template class for the solver, and then implement various specific solvers.

A solver should take in the following inputs:
 - current position
    - In x, y, z
 - current velocity
    - In x, y, z
 - current orientation
    - Just yaw
 - waypoints
    - in (x, y, z). Must be in sequence to travel

 Outputs:
    - trajectory
        - This will be a list of waypoints with an associated time
        - x , y , z , t
'''

class BaseSolver:
   def __init__(self): 
      self.current_position = None
      self.current_velocity = None
      self.current_orientation = None
      self.waypoints = None
      pass

   def _parse_waypoints(self, waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
      if not isinstance(waypoints, np.ndarray): waypoints = np.array(waypoints)
      if (waypoints[0]!=self.current_position).all() and self.current_position is not None:
         waypoints = np.insert(waypoints, 0, self.current_position, axis=0)
      x_points = waypoints[:, 0]
      y_points = waypoints[:, 1]
      z_points = waypoints[:, 2]
      return x_points, y_points, z_points

   def _solve(self): pass

   def solve(self,      current_position: np.ndarray,
                        waypoints: np.ndarray,
                        current_velocity: np.ndarray = np.zeros(3),
                        current_orientation: float = 0.0,
                        ) -> np.ndarray:
      '''
      Assumes current position is the first waypoint. Depending on the solver not all metrics may be used.
      '''
      self.current_position = current_position
      self.current_velocity = current_velocity
      self.current_orientation = current_orientation
      self.waypoints = waypoints
      return self._solve()
   
   def visualize(self, trajectory: np.ndarray, waypoints : np.ndarray) -> None:
      '''
      Solves and then visualizes the trajectory in 3D.
      '''
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
      x_points, y_points, z_points = self._parse_waypoints(waypoints)
      ax.scatter(x_points, y_points, z_points, color='red', label='Waypoints')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.legend()
      for waypoint in self.waypoints:
         ax.scatter(waypoint[0], waypoint[1], waypoint[2], c='g', marker='x')
      try:
         os.environ["DISPLAY"]
         plt.show()
      except:
         print("Unable to show plot. Saving instead...")
         plt.savefig('trajectory.png')
    
class SimpleCubicSplineSolver(BaseSolver):
   def __init__(self):
      super().__init__()

   def _solve(self) -> np.ndarray:
      '''
      Uses a parametric cubic spline to generate a smooth trajectory.
      '''
      x_points, y_points, z_points = self._parse_waypoints(self.waypoints)
      # Use arc length to parameterize the spline
      arc_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2))
      arc_length = np.insert(arc_length, 0, 0)  # offset to start from zero
      t = arc_length / arc_length[-1]  # Normalize to [0,1]
      # Solve the spline
      spline_x = CubicSpline(t, x_points, bc_type='natural')
      spline_y = CubicSpline(t, y_points, bc_type='natural')
      spline_z = CubicSpline(t, z_points, bc_type='natural')
      # Generate the trajectory
      t_fine = np.linspace(0, 1, 100)
      x_smooth = spline_x(t_fine)
      y_smooth = spline_y(t_fine)
      z_smooth = spline_z(t_fine)
      trajectory = np.array(list(zip(x_smooth, y_smooth, z_smooth, t_fine)))
      return trajectory
      
if __name__  == "__main__":
   solver = SimpleCubicSplineSolver()
   waypoints = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 2], [3, -2, 2], [1.5, 0, 2], [5, 1, 1]])
   trajectory = solver.solve(np.array([0, 0, 0]), waypoints)
   solver.visualize(trajectory, waypoints)
