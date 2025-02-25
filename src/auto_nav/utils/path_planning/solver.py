import os
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, LSQUnivariateSpline, UnivariateSpline

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

class Profile:
   def __init__(self, velocity: np.ndarray, acceleration: np.ndarray, jerk: np.ndarray, snap: np.ndarray):
      self.velocity = velocity
      self.acceleration = acceleration
      self.jerk = jerk
      self.snap = snap

class BaseSolver:
   def __init__(self): 
      self.current_position : np.ndarray = None
      self.current_velocity : np.ndarray = None
      self.current_orientation = None
      self.waypoints = None

      self.max_velocity = None
      self.max_acceleration = None
      self.max_jerk = None
      self.max_yaw_rate = None
      self.max_yaw_acceleration = None
      self.tolerance = 0.2

      self.constraints = {}

   def _parse_waypoints(self, waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
      if not isinstance(waypoints, np.ndarray): waypoints = np.array(waypoints)
      if (waypoints[0]!=self.current_position).all() and self.current_position is not None:
         waypoints = np.insert(waypoints, 0, self.current_position, axis=0)
      x_points = waypoints[:, 0]
      y_points = waypoints[:, 1]
      z_points = waypoints[:, 2]
      return x_points, y_points, z_points
   
   def set_hard_constraints(self, **kwargs):
      '''
      Set hard constraints for the solver.
      Available constraints:
         - max_velocity : float
         - max_acceleration : float
         - max_jerk : float
         - max_yaw_rate : float
         - max_yaw_acceleration : float
         - max_tolerance : float (meters) default 0.2
      '''
      self.max_velocity = kwargs.get('max_velocity', None)
      self.max_acceleration = kwargs.get('max_acceleration', None)
      self.max_jerk = kwargs.get('max_jerk', None)
      self.max_yaw_rate = kwargs.get('max_yaw_rate', None)
      self.max_yaw_acceleration = kwargs.get('max_yaw_acceleration', None)
      self.tolerance = kwargs.get("max_tolerance", 0.2) # meters
      self.constraints = kwargs

   def get_hard_constraints(self):
      return self.constraints

   def _solve(sel, **kwargs): pass

   def solve(self,      current_position: np.ndarray | None,
                        waypoints: np.ndarray,
                        current_velocity: np.ndarray | None = None,
                        current_orientation: float = 0.0,
                        **kwargs
                        ) -> np.ndarray | None:
      '''
      Assumes current position is the first waypoint. Depending on the solver not all metrics may be used.
      If the solver requires additional parameters, they can be passed as kwargs.
      '''
      if current_velocity is None:
         self.current_velocity = np.zeros(3)
      # Ensure that if the current velocity is greater than the max velocity, the max velocity is adjusted (only the greater values)
      if self.max_velocity is not None:
         for i in range(3):
            if abs(current_velocity[i]) > abs(self.max_velocity[i]):
               self.max_velocity[i] = current_velocity[i]
      self.current_position = current_position
      self.current_orientation = current_orientation
      self.waypoints = waypoints
      return self._solve(**kwargs)
   
   def profile(self, trajectory: np.ndarray) -> Profile:
      '''
      Returns a profile object that contains the velocity, acceleration, jerk, and snap profiles.
      Trajectory should be in the format of x, y, z, t.
      '''
      if trajectory is None:
         return None
      X = trajectory[:, :3].T
      T = trajectory[:, 3]
      velocity = np.gradient(X, T, axis=1)
      acceleration = np.gradient(velocity, T, axis=1)
      jerk = np.gradient(acceleration, T, axis=1)
      snap = np.gradient(jerk, T, axis=1)
      return Profile(velocity, acceleration, jerk, snap)
   
   def temporal_scale(self, trajectory: np.ndarray, max_time = None) -> np.ndarray:
      '''
      Scales the trajectory in time to meet the constraints.
      '''
      if trajectory is None:
         return None
      # Get the time from the trajectory
      time = trajectory[:, 3]
      # Iteratively scale time until all constraints are met
      for i in range(1000):
         if max_time is not None and time[-1] > max_time:
            multiplier = max_time / time[-1]
            time = time * multiplier
            trajectory[:, 3] = time
            break
         # Replace the time in the trajectory
         trajectory[:, 3] = time
         # Check constraints
         profile = self.profile(trajectory)
         if (self.max_velocity is None or np.all(np.abs(profile.velocity) <= self.max_velocity)) and \
            (self.max_acceleration is None or np.all(np.abs(profile.acceleration) <= self.max_acceleration)) and \
            (self.max_jerk is None or np.all(np.abs(profile.jerk) <= self.max_jerk)):
               break
         # Scale time
         time = time * 1.1
      return trajectory

   def visualize(self, trajectory: np.ndarray, waypoints : np.ndarray, profile : Profile = None) -> None:
      '''
      Solves and then visualizes the trajectory in 3D.
      '''
      if trajectory is None:
         return
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

      if profile is not None:
         fig1, axs = plt.subplots(3, 3, figsize=(10, 6))
         time = trajectory[:, 3]
         axs[0, 0].plot(time, profile.velocity[0], label='Velocity X')
         axs[0, 0].set_ylabel('Velocity X')
         axs[0, 1].plot(time, profile.velocity[1], label='Velocity Y')
         axs[0, 1].set_ylabel('Velocity Y')
         axs[0, 2].plot(time, profile.velocity[2], label='Velocity Z')
         axs[0, 2].set_ylabel('Velocity Z')
         axs[1, 0].plot(time, profile.acceleration[0], label='Acceleration X')
         axs[1, 0].set_ylabel('Acceleration X')
         axs[1, 1].plot(time, profile.acceleration[1], label='Acceleration Y')
         axs[1, 1].set_ylabel('Acceleration Y')
         axs[1, 2].plot(time, profile.acceleration[2], label='Acceleration Z')
         axs[1, 2].set_ylabel('Acceleration Z')
         axs[2, 0].plot(time, profile.jerk[0], label='Jerk X')
         axs[2, 0].set_ylabel('Jerk X')
         axs[2, 1].plot(time, profile.jerk[1], label='Jerk Y')
         axs[2, 1].set_ylabel('Jerk Y')
         axs[2, 2].plot(time, profile.jerk[2], label='Jerk Z')
         axs[2, 2].set_ylabel('Jerk Z')
         for ax in axs.flat:
            ax.set_xlabel('Time')
            ax.legend()
         plt.tight_layout()

      try:
         os.environ["DISPLAY"]
         plt.show()
      except:
         print("Unable to show plot. Saving instead...")
         fig.savefig('logs/trajectory.png')
         if profile is not None:
            fig1.savefig('logs/profile.png')
    
class CubicSolver(BaseSolver):
   def __init__(self):
      super().__init__()

   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Uses a parametric cubic spline to generate a smooth trajectory.
      '''
      x_points, y_points, z_points = self._parse_waypoints(self.waypoints)
      # Use euclidean length to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)  # offset to start from zero
      t = euclidean_length / euclidean_length[-1]  # Normalize to [0,1]
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
   
class LSQSolver(BaseSolver):
   def __init__(self):
      super().__init__()

   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Uses least squares to generate a smooth cubic trajectory.
      '''
      x_points, y_points, z_points = self._parse_waypoints(self.waypoints)
      # Use euclidean length to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)  # offset to start from zero
      t = euclidean_length / euclidean_length[-1]  # Normalize to [0,1]

      if("smoothing" in kwargs):
         smoothing_factor = kwargs["smoothing"]
         print("Smoothing factor: ", smoothing_factor)
      else:
         smoothing_factor = 0
      x_knots = UnivariateSpline(t, x_points, k=3, s=smoothing_factor).get_knots()[1:-1]
      y_knots = UnivariateSpline(t, y_points, k=3, s=smoothing_factor).get_knots()[1:-1]
      z_knots = UnivariateSpline(t, z_points, k=3, s=smoothing_factor).get_knots()[1:-1]

      spline_x = LSQUnivariateSpline(t, x_points, x_knots)
      spline_y = LSQUnivariateSpline(t, y_points, y_knots)
      spline_z = LSQUnivariateSpline(t, z_points, z_knots)
      # Generate the trajectory
      t_fine = np.linspace(0, 1, 100)
      x_smooth = spline_x(t_fine)
      y_smooth = spline_y(t_fine)
      z_smooth = spline_z(t_fine)
      trajectory = np.array(list(zip(x_smooth, y_smooth, z_smooth, t_fine)))
      return trajectory
   
class QPSolver(BaseSolver):
   def __init__(self):
      super().__init__()
   
   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Formats the problem into a QP and solves it.
      Minimizes snap.
      '''
      x_points, y_points, z_points = self._parse_waypoints(self.waypoints)
      # Use euclidean dist to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)

      #--- FORMULATE QP PROBLEM ---#
      # Use max distance to approximate time based on 2 m/s avg speed
      dt = 0.1
      waypoint_times = np.rint((euclidean_length / 2) / dt).astype(int)
      T = waypoint_times[-1] + 1
      #--- Define optimization variables ---#
      X = cp.Variable((3, T))  # Position
      V = cp.Variable((3, T))  # Velocity
      A = cp.Variable((3, T))  # Acceleration
      J = cp.Variable((3, T))  # Jerk
      S = cp.Variable((3, T))  # Snap
      #--- Define initial conditions ---#
      x0, y0, z0 = x_points[0], y_points[0], z_points[0]
      constraints = [X[:, 0] == np.array([x0, y0, z0])]
      constraints += [V[:, 0] == self.current_velocity]
      #--- Add motion constraints ---#
      for t in range(T - 1):
         constraints += [
            X[:, t+1] == X[:, t] + V[:, t] * dt + 0.5 * A[:, t] * dt**2 + (1/6) * J[:, t] * dt**3 + (1/24) * S[:, t] * dt**4,
            V[:, t+1] == V[:, t] + A[:, t] * dt + 0.5 * J[:, t] * dt**2 + (1/6) * S[:, t] * dt**3,
            A[:, t+1] == A[:, t] + J[:, t] * dt + 0.5 * S[:, t] * dt**2,
            J[:, t+1] == J[:, t] + S[:, t] * dt,
         ]
      #--- Velocity and acceleration limits ---#
      if self.max_velocity is not None:
         for i in range(3):
            constraints += [cp.abs(V[i, :]) <= self.max_velocity]
      if self.max_acceleration is not None:
         for i in range(3):
            constraints += [cp.abs(A[i, :]) <= self.max_acceleration]
      if self.max_jerk is not None:
         for i in range(3):
            constraints += [cp.abs(J[i, :]) <= self.max_jerk]
      #--- Define waypoints and tolerance ---#
      tolerance = self.tolerance
      for i, t_idx in enumerate(waypoint_times):
         constraints.append(X[:, int(t_idx)] >= np.array([x_points[i] - tolerance, y_points[i] - tolerance, z_points[i] - tolerance]))
         constraints.append(X[:, int(t_idx)] <= np.array([x_points[i] + tolerance, y_points[i] + tolerance, z_points[i] + tolerance]))
      #--- Define cost function (acceleration, jerk, snap) ---#
      cost = cp.sum_squares(A)
      cost += cp.sum_squares(J)
      cost += cp.sum_squares(S)*100 # penalize snap more
      #--- Solve the optimization problem ---#
      problem = cp.Problem(cp.Minimize(cost), constraints)
      try:
         problem.solve(solver=cp.OSQP, verbose=False, max_iter=20000)
      except cp.SolverError as e:
         print("Failed to solve")
         return None
      #--- Extract optimized trajectory ---#
      if X.value is None:
         print("Failed to solve")
         return None
      trajectory = X.value.T
      trajectory = np.insert(trajectory, 3, np.linspace(0, T*dt, T), axis=1)
      return trajectory


      
if __name__  == "__main__":
   solver = QPSolver()
   waypoints = np.array([  [0, 0, 0], 
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
   solver.set_hard_constraints(max_tolerance=0.2)
   trajectory = solver.solve(None, waypoints)
   profile = solver.profile(trajectory)
   solver.visualize(trajectory, waypoints, profile)

   solver.set_hard_constraints(max_jerk=3)
   solver.temporal_scale(trajectory)
   profile = solver.profile(trajectory)
   solver.visualize(trajectory, waypoints, profile)
