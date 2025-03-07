import os
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cvxpy as cp
import casadi as ca
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

class CasSolver(BaseSolver):
   def __init__(self):
      super().__init__()
   
   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Formats the problem into a constraint problem and solves it using ipopt.
      Minimizes snap and attempts yaw.
      '''
      x_points, y_points, z_points = self._parse_waypoints(self.waypoints)
      # Use euclidean dist to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)

      #--- FORMULATE CONSTRAINT PROBLEM ---#
      # Use max distance to approximate time based on 2 m/s avg speed
      dt = 0.1
      waypoint_times = np.rint((euclidean_length / 2) / dt).astype(int)
      T = waypoint_times[-1] + 1
      #--- Define optimization variables ---#
      optimizer = ca.Opti()
      X = optimizer.variable(3, T)  # Position
      V = optimizer.variable(3, T)  # Velocity
      A = optimizer.variable(3, T)  # Acceleration
      J = optimizer.variable(3, T)  # Jerk
      S = optimizer.variable(3, T)  # Snap
      psi = optimizer.variable(T) # Yaw
      psi_dot = optimizer.variable(T) # Yaw rate
      psi_ddot = optimizer.variable(T) # Yaw acceleration
      #--- Define initial conditions ---#
      x0, y0, z0 = x_points[0], y_points[0], z_points[0]
      optimizer.subject_to(X[:, 0] == np.array([x0, y0, z0]))
      if(np.all(self.current_velocity == 0)):
         self.current_velocity = np.array([1e-3, 0, 0])
      optimizer.subject_to(V[:, 0] == self.current_velocity)
      optimizer.subject_to(psi[0] == self.current_orientation)
      #--- Add motion constraints ---#
      for t in range(T - 1):
         R_t = ca.vertcat(
            ca.horzcat(ca.cos(psi[t]), -ca.sin(psi[t]), 0),
            ca.horzcat(ca.sin(psi[t]), ca.cos(psi[t]), 0),
            ca.horzcat(0, 0, 1)
         )
         optimizer.subject_to(X[:, t+1] == X[:, t] + R_t @ V[:, t] * dt + 0.5 * R_t @ A[:, t] * dt**2 + (1/6) * R_t @ J[:, t] * dt**3 + (1/24) * R_t @ S[:, t] * dt**4)
         optimizer.subject_to(V[:, t+1] == V[:, t] + A[:, t] * dt + 0.5 * J[:, t] * dt**2 + (1/6) * S[:, t] * dt**3)
         optimizer.subject_to(A[:, t+1] == A[:, t] + J[:, t] * dt + 0.5 * S[:, t] * dt**2)
         optimizer.subject_to(J[:, t+1] == J[:, t] + S[:, t] * dt)

         optimizer.subject_to(psi[t+1] == psi[t] + psi_dot[t] * dt + 0.5 * psi_ddot[t] * dt**2)
         optimizer.subject_to(psi_dot[t+1] == psi_dot[t] + psi_ddot[t] * dt)
      #--- Velocity and acceleration limits ---#
      if self.max_velocity is not None:
         optimizer.subject_to(ca.norm_2(V) <= self.max_velocity)
      if self.max_acceleration is not None:
         optimizer.subject_to(ca.norm_2(A) <= self.max_acceleration)             
      if self.max_jerk is not None:
         optimizer.subject_to(ca.norm_2(J) <= self.max_jerk)
      #--- Define waypoints and tolerance ---#
      tolerance = self.tolerance
      for i, t_idx in enumerate(waypoint_times):
         optimizer.subject_to(X[:, int(t_idx)] >= np.array([x_points[i] - tolerance, y_points[i] - tolerance, z_points[i] - tolerance]))
         optimizer.subject_to(X[:, int(t_idx)] <= np.array([x_points[i] + tolerance, y_points[i] + tolerance, z_points[i] + tolerance]))
      #--- Define cost function (acceleration, jerk, snap) ---#
      cost = ca.sumsqr(A)
      cost += ca.sumsqr(J)
      cost += ca.sumsqr(S)
      #--- Yaw cost ---#
      eps = 1e-6
      heading_angle = ca.atan2(V[1, :] + eps, V[0, :] + eps)
      cost += ca.sumsqr(psi - ca.transpose(heading_angle)) * 10
      cost+= ca.sumsqr(psi_dot)
      cost+= ca.sumsqr(psi_ddot)
      #--- Solve the optimization problem ---#
      optimizer.minimize(cost)
      opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
      optimizer.solver('ipopt', opts)
      solution : ca.OptiSol = None
      try:
         solution = optimizer.solve()
      except:
         print("Failed to solve")
         return None
      #--- Extract optimized trajectory ---#
      if solution.value(X) is None:
         print("Failed to solve")
         return None
      trajectory = solution.value(X)
      trajectory = trajectory.T
      yaw = solution.value(psi)
      yaw = yaw.T
      trajectory = np.insert(trajectory, 3, np.linspace(0, T*dt, T), axis=1)
      # insert yaw as the last column
      trajectory = np.insert(trajectory, 4, yaw, axis=1)
      return trajectory


      
if __name__  == "__main__":
   solver = CasSolver()
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
#    waypoints = np.array([[  1.21      ,  10.24      ,   1.35],
#  [  2.92606891,  12.44012771,   1.35],
#  [  4.37393109,  13.81987229,   1.35],
#  [  7.74597775,  11.99999738,   1.35],
#  [ 12.57402225,  10.70000262,   1.35],
#  [ 14.11419013,  12.93208394,   1.35],
#  [ 16.47548592,  16.16074963,   1.35],
#  [ 16.47548592,  16.16074963,   1.45],
#  [ 17.25158047,  16.0631713 ,   1.75],
#  [ 17.73321315,  15.47721697,   2.05],
#  [ 17.94227549,  14.69529186,   2.35],
#  [ 17.83353107,  13.84701844,   2.65],
#  [ 17.40271384,  13.06103983,   2.95],
#  [ 16.69042814,  12.45735231,   3.25],
#  [ 15.77806617,  12.133396  ,   3.55],
#  [ 14.77793725,  12.15121548,   3.85],
#  [ 13.81902815,  12.52850073,   4.15],
#  [ 14.11419013,  12.93208394,   4.05],
#  [ 16.18032395,  15.75716642,   4.05],
#  [ 18.18032395,  15.75716642,   4.05],
#  [ 18.77499031,  15.88005632,   4.05],
#  [ 18.80500969,  19.87994368,   4.05],
#  [ 18.80500969,  19.87994368,   1.35],
#  [ 18.78249515,  16.88002816,   1.35],
#  [ 16.7706479 ,  16.56433285,   1.35],
#  [ 14.4093521 ,  13.33566715,   1.35],
#  [ 14.81902815,  12.52850073,   1.35],
#  [ 16.04999235,  11.98547524,   1.45],
#  [ 16.80048596,  12.20805355,   1.75],
#  [ 17.45686885,  12.62974244,   2.05],
#  [ 17.95170916,  13.22275585,   2.35],
#  [ 18.22362757,  13.92954066,   2.65],
#  [ 18.23162321,  14.66396083,   2.95],
#  [ 17.96893962,  15.31912966,   3.25],
#  [ 17.473598  ,  15.78155123,   3.55],
#  [ 16.83295695,  15.9500776 ,   3.85],
#  [ 16.18032395,  15.75716642,   4.15],
#  [ 16.18032395,  15.75716642,   4.05],
#  [ 14.4093521 ,  13.33566715,   4.05],
#  [ 12.0912178 ,  10.83000209,   1.35],
#  [  9.1943911 ,  11.60999895,   1.35],
#  [  8.2287822 ,  10.86999791,   1.35],
#  [  1.84017226,   9.40531929,   1.35],
#  [  1.21      ,  10.24      ,   1.  ]])
   # waypoints = np.delete(waypoints, 3, axis=1)
   solver.set_hard_constraints(max_tolerance=0.2)
   trajectory = solver.solve(None, waypoints)
   profile = solver.profile(trajectory)
   solver.visualize(trajectory, waypoints, profile)

   # solver.set_hard_constraints(max_jerk=3)
   # solver.temporal_scale(trajectory)
   # profile = solver.profile(trajectory)
   # solver.visualize(trajectory, waypoints, profile)
