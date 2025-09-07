import os
import time as ti
import warnings
warnings.simplefilter("ignore", UserWarning)
import math
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
   def __init__(self, times: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, jerk: np.ndarray, snap: np.ndarray, 
                psi: np.ndarray = None, psi_dot: np.ndarray = None, psi_ddot: np.ndarray = None, body_velocity: np.ndarray = None):
      self.times = times
      self.velocity = velocity
      self.body_velocity = body_velocity
      self.acceleration = acceleration
      self.jerk = jerk
      self.snap = snap
      self.psi = psi
      self.psi_dot = psi_dot
      self.psi_ddot = psi_ddot

      self._actual_path : np.ndarray = None

   def get_velocity(self) -> np.ndarray:
      '''
      Returns the velocity profile as vx, vy, vz, t, yr
      If there is no yaw rate, it will return vx, vy, vz, t
      '''
      velocity = self.velocity.T
      velocity = np.insert(velocity, 3, self.times, axis=1)
      if self.psi is not None:
         velocity = np.insert(velocity, 4, self.psi_dot, axis=1)
      return velocity
   
   def get_acceleration(self) -> np.ndarray:
      '''
      Returns the acceleration profile as ax, ay, az, t, yrr
      If there is no yaw rate, it will return ax, ay, az, t
      '''
      acceleration = self.acceleration.T
      acceleration = np.insert(acceleration, 3, self.times, axis=1)
      if self.psi is not None:
         acceleration = np.insert(acceleration, 4, self.psi_ddot, axis=1)
      return acceleration
   
   def save_point(self, point: np.ndarray) -> None:
      '''
      Saves a point to the actual path.
      '''
      if self._actual_path is None:
         self._actual_path = point
      else:
         self._actual_path = np.vstack((self._actual_path, point))

   def clear_actual_path(self) -> None:
      self._actual_path = None

   def get_actual_path(self) -> np.ndarray:
      return self._actual_path
      
class BaseSolver:
   def __init__(self): 
      self.current_position : np.ndarray = None
      self.current_velocity : np.ndarray = [0,0,0]
      self.current_orientation = None
      self.waypoints = None

      self.max_velocity = None
      self.max_acceleration = None
      self.max_jerk = None
      self.max_yaw_rate = None
      self.max_yaw_acceleration = None
      self.tolerance = 0.2

      self.constraints = {}

   def _parse_waypoints(self, waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
      if not isinstance(waypoints, np.ndarray): waypoints = np.array(waypoints)
      if not np.array_equal(waypoints[0],self.current_position) and self.current_position is not None:
         if waypoints.shape[1] == 4: pose = np.append(self.current_position, self.current_orientation)
         else: pose = self.current_position
         waypoints = np.insert(waypoints, 0, pose, axis=0)
      x_points = waypoints[:, 0]
      y_points = waypoints[:, 1]
      z_points = waypoints[:, 2]
      yaw_points = None
      if waypoints.shape[1] == 4:
         if self.current_orientation is not None:
            waypoints[0, 3] = self.current_orientation
         yaw_points = waypoints[:, 3]
      return x_points, y_points, z_points, yaw_points
   
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
                        current_orientation: float | None = None,
                        **kwargs
                        ) -> np.ndarray | None:
      '''
      Assumes current position is the first waypoint. Depending on the solver not all metrics may be used.
      If the solver requires additional parameters, they can be passed as kwargs.
      '''
      if current_velocity is None:
         current_velocity = np.zeros(3)
      if len(current_velocity) == 4:
         current_velocity = current_velocity[:3]
      # Ensure that if the current velocity is greater than the max velocity, the max velocity is adjusted (only the greater values)
      if self.max_velocity is not None:
         for i in range(3):
            if abs(current_velocity[i]) > abs(self.max_velocity):
               self.max_velocity = current_velocity[i]
      self.current_position = current_position
      self.current_velocity = current_velocity
      self.current_orientation = current_orientation
      self.waypoints = waypoints
      return self._solve(**kwargs)
   
   def profile(self, trajectory: np.ndarray, *, use_body = False) -> Profile:
      '''
      Returns a profile object that contains the velocity, acceleration, jerk, and snap profiles.
      Trajectory should be in the format of x, y, z, t.
      '''
      if trajectory is None:
         return None
      X = trajectory[:, :3].T
      T = trajectory[:, 3]
      bv = None
      if trajectory.shape[1] > 4:
         bv = trajectory[:, 5:8].T
      if use_body and bv is not None:
         velocity = bv
      else:
         velocity = np.gradient(X, T, axis=1)
      acceleration = np.gradient(velocity, T, axis=1)
      jerk = np.gradient(acceleration, T, axis=1)
      snap = np.gradient(jerk, T, axis=1)
      if trajectory.shape[1] > 4:
         yaw = trajectory[:, 4]
         yaw_dot = np.gradient(yaw, T)
         yaw_ddot = np.gradient(yaw_dot, T)
         return Profile(T, velocity, acceleration, jerk, snap, yaw, yaw_dot, yaw_ddot, bv)
      return Profile(T, velocity, acceleration, jerk, snap)
   
   def temporal_scale(self, trajectory: np.ndarray, *, set_time = None, max_time = None) -> np.ndarray:
      '''
      Scales the trajectory in time to meet the constraints.
      DOES NOT WORK FOR CASADI SOLVER----
      '''
      if trajectory is None:
         return None
      if set_time is not None:
         time_var = trajectory[:, 3]
         multiplier = set_time / time_var[-1]
         time_var = time_var * multiplier
         trajectory[:, 3] = time_var
         return trajectory
      # if trajectory.shape[1] > 4:
      #    print("Temporal scaling not implemented for casadi solver.")
      #    return trajectory
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

   def visualize(self, trajectory: np.ndarray, waypoints : np.ndarray = None, profile : Profile = None, *, actual_traj : np.ndarray = None) -> None:
      '''
      Solves and then visualizes the trajectory in 3D.
      '''
      if trajectory is None:
         return
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.set_zlim(0,7)
      ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
      # If an actual path is provided, plot it in blue
      if actual_traj is not None:
         ax.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2], color='blue', label='Actual Path')
      if waypoints is not None:
         x_points, y_points, z_points, yaw_points = self._parse_waypoints(waypoints)
         ax.scatter(x_points, y_points, z_points, color='red', label='Waypoints')
      if self.waypoints is not None:
         for waypoint in self.waypoints:
            ax.scatter(waypoint[0], waypoint[1], waypoint[2], c='g', marker='x')
      # Draw arrow for orientation if available
      if trajectory.shape[1] > 4:
         yaw = trajectory[:, 4]
         for i in range(len(trajectory)):
            ax.quiver(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2], np.cos(yaw[i]), np.sin(yaw[i]), 0, length=0.5, normalize=True)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.legend()

      if profile is not None:
         fig1, axs = plt.subplots(3, 3+(isinstance(profile.psi, np.ndarray)), figsize=(10, 6))
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
         if isinstance(profile.psi, np.ndarray):
            axs[0, 3].plot(time, profile.psi, label='Yaw')
            axs[0, 3].set_ylabel('Yaw')
            axs[1, 3].plot(time, profile.psi_dot, label='Yaw Rate')
            axs[1, 3].set_ylabel('Yaw Rate')
            axs[2, 3].plot(time, profile.psi_ddot, label='Yaw Acceleration')
            axs[2, 3].set_ylabel('Yaw Acceleration')
         for ax in axs.flat:
            ax.set_xlabel('Time')
            ax.legend()
         plt.tight_layout()

      try:
         os.environ["DISPLAY"]
         plt.show()
      except:
         print("Unable to show plot. Saving instead...")
         date_timestamp = ti.strftime('%Y_%m_%d-%H_%M_%S')
         if not os.path.exists('logs'):
            os.makedirs('logs')
         actual = '_actual' if actual_traj is not None else ''
         fig.savefig(f'logs/trajectory_{date_timestamp}{actual}.png')
         if profile is not None:
            fig1.savefig(f'logs/profile_{date_timestamp}{actual}.png')
    
class CubicSolver(BaseSolver):
   def __init__(self):
      super().__init__()

   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Uses a parametric cubic spline to generate a smooth trajectory.
      '''
      x_points, y_points, z_points, yaw_points = self._parse_waypoints(self.waypoints)
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
      x_points, y_points, z_points, yaw_points = self._parse_waypoints(self.waypoints)
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
      x_points, y_points, z_points, yaw_points = self._parse_waypoints(self.waypoints)
      # Use euclidean dist to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)

      #--- FORMULATE QP PROBLEM ---#
      # Use max distance to approximate time based on 2 m/s avg speed
      dt = 0.05
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
      yaw_tolerance = 2*(np.pi/180)
      for i, t_idx in enumerate(waypoint_times):
         constraints.append(X[:, int(t_idx)] >= np.array([x_points[i] - tolerance, y_points[i] - tolerance, z_points[i] - tolerance]))
         constraints.append(X[:, int(t_idx)] <= np.array([x_points[i] + tolerance, y_points[i] + tolerance, z_points[i] + tolerance]))
      #--- Define cost function (acceleration, jerk, snap) ---#
      cost = cp.sum_squares(A)
      cost += cp.sum_squares(J)
      cost += cp.sum_squares(S)*10 # penalize snap more
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
      x_points, y_points, z_points, yaw_points = self._parse_waypoints(self.waypoints)
      # Use euclidean dist to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)
      #--- FORMULATE CONSTRAINT PROBLEM ---#
      # Use max distance to approximate time based on 2 m/s avg speed
      dt = 0.05
      waypoint_times = np.rint((euclidean_length / kwargs.get("desired_velocity", 2)) / dt).astype(int)
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
      if yaw_points is not None:
         optimizer.subject_to(psi[0] == yaw_points[0] * (np.pi/180))
      #--- Add motion constraints ---#
      for t in range(T - 1):
         if(kwargs.get("use_body_vel", False)):
            R_t = ca.vertcat(
               ca.horzcat(ca.cos(psi[t]), -ca.sin(psi[t]), 0),
               ca.horzcat(ca.sin(psi[t]), ca.cos(psi[t]), 0),
               ca.horzcat(0, 0, 1)
            )
            optimizer.subject_to(X[:, t+1] == X[:, t] + R_t @ V[:, t] * dt + 0.5 * R_t @ A[:, t] * dt**2 + (1/6) * R_t @ J[:, t] * dt**3 + (1/24) * R_t @ S[:, t] * dt**4)
         else:
            optimizer.subject_to(X[:, t+1] == X[:, t] + V[:, t] * dt + 0.5 * A[:, t] * dt**2 + (1/6) * J[:, t] * dt**3 + (1/24) * S[:, t] * dt**4)
         optimizer.subject_to(V[:, t+1] == V[:, t] + A[:, t] * dt + 0.5 * J[:, t] * dt**2 + (1/6) * S[:, t] * dt**3)
         optimizer.subject_to(A[:, t+1] == A[:, t] + J[:, t] * dt + 0.5 * S[:, t] * dt**2)
         optimizer.subject_to(J[:, t+1] == J[:, t] + S[:, t] * dt)

         optimizer.subject_to(psi[t+1] == psi[t] + psi_dot[t] * dt + 0.5 * psi_ddot[t] * dt**2)
         optimizer.subject_to(psi_dot[t+1] == psi_dot[t] + psi_ddot[t] * dt)
      #--- Velocity and acceleration limits ---#
      # Pos Contraints #TODO: Implement corrdior constraints
      if kwargs.get("min_height", None) is not None:
         print("Min Height set: ", kwargs.get("min_height"))
         optimizer.subject_to(X[2,:] >= kwargs.get("min_height"))
      if kwargs.get("slow_at_end", False):
         print("Slow at end set")
         for i in range(3):
            optimizer.subject_to(V[i, -1] <= 0.3)
            optimizer.subject_to(V[i, -1] >= -0.3)

      # Velocity constraints
      if self.max_velocity is not None:
         for i in range(3):
            optimizer.subject_to(-self.max_velocity <= V[i, :])
            optimizer.subject_to(V[i, :] <= self.max_velocity)

      # Acceleration constraints
      if self.max_acceleration is not None:
         for i in range(3):
            optimizer.subject_to(-self.max_acceleration <= A[i, :])
            optimizer.subject_to(A[i, :] <= self.max_acceleration)

      # Jerk constraints
      if self.max_jerk is not None:
         for i in range(3):
            optimizer.subject_to(-self.max_jerk <= J[i, :])
            optimizer.subject_to(J[i, :] <= self.max_jerk)

      #--- Define waypoints and tolerance ---#
      tolerance = self.tolerance
      yaw_tolerance = 2*(np.pi/180) # to radians
      for i, t_idx in enumerate(waypoint_times):
         optimizer.subject_to(X[:, int(t_idx)] >= np.array([x_points[i] - tolerance, y_points[i] - tolerance, z_points[i] - tolerance]))
         optimizer.subject_to(X[:, int(t_idx)] <= np.array([x_points[i] + tolerance, y_points[i] + tolerance, z_points[i] + tolerance]))
         if yaw_points is not None:
            if yaw_points[i] != -1:
               target_yaw = yaw_points[i] * (np.pi/180)
               angle_diff = ca.fmod(psi[int(t_idx)] - target_yaw + np.pi, 2*np.pi) - np.pi
               optimizer.subject_to(ca.fabs(angle_diff) <= yaw_tolerance)
      #--- Define cost function (acceleration, jerk, snap) ---#
      cost = ca.sumsqr(A)
      cost += ca.sumsqr(J)
      cost += ca.sumsqr(S)
      #--- Yaw cost ---#
      # eps = 1e-6
      # heading_angle = ca.atan2(V[1, :] + eps, V[0, :] + eps)
      # cost += ca.sumsqr(psi - ca.transpose(heading_angle))
      cost += ca.sumsqr(psi_dot)
      cost += ca.sumsqr(psi_ddot)
      #--- Solve the optimization problem ---#
      optimizer.minimize(cost)
      opts = {}
      opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
      optimizer.solver('ipopt', opts)
      solution : ca.OptiSol = None
      try:
         solution = optimizer.solve()
      except Exception as e:
         print(f"Failed to solve: \n{e}")
         return None
      #--- Extract optimized trajectory ---#
      if solution.value(X) is None:
         print("Failed to solve: X is None")
         return None
      trajectory = solution.value(X)
      trajectory = trajectory.T
      velocity = solution.value(V)
      velocity = velocity.T
      yaw = solution.value(psi)
      yaw = yaw.T
      trajectory = np.insert(trajectory, 3, np.linspace(0, T*dt, T), axis=1)
      trajectory = np.insert(trajectory, 4, yaw, axis=1)
      trajectory = np.concatenate((trajectory, velocity), axis=1)
      return trajectory


class QPSolver2(BaseSolver):
   def __init__(self):
      super().__init__()

   def _form_basis(self, t: float, deg: int = 7) -> np.ndarray:
      return np.array([t**i for i in range(deg+1)])
   
   def _get_coeffs(self, order: int, deg: int = 7) -> np.ndarray:
      coeffs = np.zeros(deg+1)
      for i in range(order, deg+1):
         coeffs[i] = math.factorial(i) / math.factorial(i - order)
      return coeffs
   
   def _get_dconstraint_row(self, t: float, order: int, deg: int = 7) -> np.ndarray:
      '''
      Returns a row vector for the constraint matrix.
      Row corresponds to the derivative order at time t.
      '''
      row = np.zeros(deg+1)
      coeffs = self._get_coeffs(order, deg)
      for i in range(order, deg+1):
         row[i] = coeffs[i] * (t ** (i - order))
      return row
   
   def _hessian_block(self, dt: float, deg: int = 7) -> np.ndarray:
      '''
      Hessian per segment for snap minimization
      Qi,j = ∫ (coeff_i * coeff_j * t^(i+j-8)) dt from t0 to tf
      Analytically solved as:
         Qi,j = (coeff_i * coeff_j / (i + j - 7)) * (tf^(i+j-7) - t0^(i+j-7))
      '''
      Q = np.zeros((deg+1, deg+1))
      snap_coeffs = self._get_coeffs(4, deg)
      for i in range(4, deg+1):
         for j in range(4, deg+1):
            Q[i, j] = (snap_coeffs[i] * snap_coeffs[j] / (i + j - 7)) * (dt**(i + j - 7))
      return Q
   
   def _build_hessian(self, segment_times: List[float], deg: int = 7) -> np.ndarray:
      '''
      Builds the full Hessian matrix for all segments.
      '''
      n_segments = len(segment_times)
      # Per segment, we have (deg + 1) coefficients
      H = np.zeros((n_segments * (deg + 1), n_segments * (deg + 1)))
      for i, dt in enumerate(segment_times):
         Q_k = self._hessian_block(dt, deg)
         # Place Q_k in the appropriate block of H (along the diagonal)
         H[i*(deg+1):(i+1)*(deg+1), i*(deg+1):(i+1)*(deg+1)] = Q_k
      return H
   
   def _build_AB_constraints(self, segment_times : List[float], waypoints : List[float], axis : int, just_b = False, deg: int = 7) -> Tuple[np.ndarray, np.ndarray]:
      '''
      Returns a matrix A and vector b such that Ax = b represents the equality constraints. (where x is the vector of all polynomial coefficients)
      Need to enforce:
         - Position constraints at waypoints
         - Continuity of velocity, acceleration, jerk at segment boundaries (position inherited from above)
         - Initial & Final conditions (velocity)
      For R3 same A can be used, just b changes.
      ''' 
      n_segments = len(segment_times)
      n_coeffs = deg + 1
      total_coeffs = n_segments * n_coeffs
      A, b = [], []
      # Position constraints at waypoints
      for i in range(n_segments):
         # Start of segment (t=0)
         if(not just_b):
            row = np.zeros(total_coeffs)
            row[i*n_coeffs:(i+1)*n_coeffs] = self._get_dconstraint_row(0.0, 0, deg)
            A.append(row)
         b.append(waypoints[i])

         # End of segment (t=dt)
         if(not just_b):
            row = np.zeros(total_coeffs)
            row[i*n_coeffs:(i+1)*n_coeffs] = self._get_dconstraint_row(segment_times[i], 0, deg)
            A.append(row)
         b.append(waypoints[i+1])
      # Continuity constraints at segment boundaries (velocity, acceleration, jerk)
      for i in range(1, n_segments):
         for order in range(1, 4):  # vel, accel, jerk
            if(just_b):
               b.append(0.0)
               continue
            row = np.zeros(total_coeffs)
            # End of previous segment
            row[(i-1)*n_coeffs:i*n_coeffs] = self._get_dconstraint_row(segment_times[i-1], order, deg)
            # Start of current segment
            row[i*n_coeffs:(i+1)*n_coeffs] -= self._get_dconstraint_row(0.0, order, deg)
            A.append(row)
            b.append(0.0)
      # Initial conditions (velocity)
      row = np.zeros(total_coeffs)
      row[0:n_coeffs] = self._get_dconstraint_row(0.0, 1, deg)
      if(not just_b): A.append(row)
      if axis < 3:
         b.append(self.current_velocity[axis])
      else:
         b.append(0.0)
      if(not just_b): A = np.vstack(A) # constraint matrix
      b = np.array(b) # constraint vector
      return A, b
   
   def _solve_qp(self, H : np.ndarray, A : np.ndarray, b : np.ndarray) -> np.ndarray:
      '''
      KKT QP Solver
      min (1/2) c^T H c
      s.t. A c = b
      '''
      n_vars = H.shape[0]
      n_constraints = A.shape[0]
      # Build KKT matrix
      KKT = np.block([
         [H, A.T],
         [A, np.zeros((n_constraints, n_constraints))]
      ])
      rhs = np.concatenate([np.zeros(n_vars), b])
      # Solve KKT system
      sol = np.linalg.solve(KKT, rhs)
      c = sol[:n_vars]  # polynomial coefficients
      return c

   def _solve_axis(self, segment_times: List[float], waypoints: List[float], axis: int, deg: int = 7) -> np.ndarray:
      '''
      Solves for the polynomial coefficients for a single axis.
      '''
      H = self._build_hessian(segment_times, deg)
      A, b = self._build_AB_constraints(segment_times, waypoints, axis, 0, deg)
      coeffs : np.ndarray = self._solve_qp(H, A, b)
      return coeffs.reshape(-1, deg + 1)
   
   def _evaluate_cost(self, segment_times: List[float], waypoints: Tuple) -> Tuple[dict, float]:
      '''
      Solves for all axes and evaluates the total cost.
      Segment times is the dt per segment. 
      Waypoints is a tuple of (x_points, y_points, z_points, yaw_points)
      Returns the coefficients for all axes and the total cost.
      '''
      rH = self._build_hessian(segment_times, 7)
      A, b = self._build_AB_constraints(segment_times, waypoints[0], 0)
      muJ = 0.5
      muPsi = 0.1
      total_cost = 0.0
      all_coeffs = {}
      for i,axis in enumerate(["x","y","z"]):
         _, b = self._build_AB_constraints(segment_times, waypoints[i], i, just_b=True, deg=7)
         coeffs = self._solve_qp(rH, A, b)
         # 1 * 1x24 * 24x24 * 24x1 = 1x1
         total_cost += muJ * (coeffs.T @ rH @ coeffs)
         all_coeffs[axis] = coeffs.reshape(-1, 8)

      H_psi = self._build_hessian(segment_times, 5)
      A_psi, b_psi = self._build_AB_constraints(segment_times, waypoints[3], 3, deg=5)
      coeffs_psi = self._solve_qp(H_psi, A_psi, b_psi)
      total_cost += muPsi * (coeffs_psi.T @ H_psi @ coeffs_psi)
      all_coeffs["yaw"] = coeffs_psi.reshape(-1, 6)
      return all_coeffs, total_cost
   
   def _build_gi(self, i, m):
      gi = np.full(m, -1/(m-2))
      gi[i] = 1.0
      return gi

   def _optimize_segment_times(self, dt: np.ndarray, waypoints: Tuple, max_iter: int = 10) -> Tuple[dict, np.ndarray]:
      h=1e-3
      lr = 0.1
      alpha = 0.4
      tolerance = 1e-6
      total_time = sum(dt)
      for it in range(max_iter):
         coeffs, cost = self._evaluate_cost(dt, waypoints)
         grad = np.zeros_like(dt)
         for i in range(len(dt)):
            gi = self._build_gi(i, len(dt))
            grad[i] = (self._evaluate_cost(dt + h*gi, waypoints)[1] - cost) / h
         dt_new = dt - lr * grad
         dt_new = np.clip(dt_new, 1e-3, None)
         dt_new = (dt_new / sum(dt_new)) * total_time # normalize
         coeffs, new_cost = self._evaluate_cost(dt_new, waypoints)
         # Backtracking line search
         while new_cost > cost and lr > 1e-6:
            lr *= alpha
            dt_new = dt - lr * grad
            dt_new = np.clip(dt_new, 1e-3, None)
            dt_new = (dt_new / sum(dt_new)) * total_time
            coeffs, new_cost = self._evaluate_cost(dt_new, waypoints)
         dt = dt_new
         if abs(new_cost - cost) < tolerance:
            print(f"Converged after {it} iterations")
            break
      return coeffs, dt
   
   def _one_axis_sample(self, coeffs, dt_list, sample_dt=0.1): # Test function for debugging
      total_time = sum(dt_list)
      global_times = np.arange(0, total_time + sample_dt, sample_dt)
      positions, velocities, accelerations = [], [], []
      seg_edges = np.cumsum([0.0] + dt_list)
      for t in global_times:
         # Find which segment this time belongs to
         seg_idx = np.searchsorted(seg_edges, t, side='right') - 1
         seg_idx = min(seg_idx, len(dt_list) - 1)  # Clamp to last segment
         local_t = t - seg_edges[seg_idx]

         c = coeffs[seg_idx]
         pos = np.polyval(c[::-1], local_t)
         # pos vel computed here for testing for now TODO: remove
         vel = np.polyval(np.polyder(c[::-1], 1), local_t)
         acc = np.polyval(np.polyder(c[::-1], 2), local_t)
         positions.append(pos)
         velocities.append(vel)
         accelerations.append(acc)
      return global_times, np.array(positions), np.array(velocities), np.array(accelerations)

   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Using the fact that each axis can be decoupled, we can solve 4 independent QPs for [x,y,z,yaw].
      Minimizes snap^2.
      '''
      x_points, y_points, z_points, yaw_points = self._parse_waypoints(self.waypoints)
      yaw_points = yaw_points if yaw_points is not None else np.zeros(len(x_points))
      yaw_points = np.unwrap(np.radians(yaw_points))
      # Use euclidean dist to parameterize the spline
      euclidean_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2 + np.diff(z_points)**2))
      euclidean_length = np.insert(euclidean_length, 0, 0)
      # Initial guess for segment times based on distance and desired velocity
      desired_velocity = kwargs.get("desired_velocity", 2.0)
      dt = (euclidean_length[1:] - euclidean_length[:-1]) / desired_velocity
      dt = np.clip(dt, 1.0, None) # min segment time of 1s
      waypoints = (x_points, y_points, z_points, yaw_points)
      if(kwargs.get("optimize", False)):
         coeffs, dt = self._optimize_segment_times(dt, waypoints, max_iter=kwargs.get("max_iter", 20))
      else:
         coeffs, _ = self._evaluate_cost(dt, waypoints)
      # Sample
      sample_dt = 0.1
      total_time = sum(dt)
      global_times = np.arange(0, total_time + sample_dt, sample_dt)
      seg_edges = np.cumsum(np.concatenate(([0.0],dt)))
      traj = np.zeros((len(global_times), 5)) # [x,y,z,time,yaw]
      for i,t in enumerate(global_times):
         # Find which segment this time belongs to
         seg_idx = np.searchsorted(seg_edges, t, side='right') - 1
         seg_idx = min(seg_idx, len(dt) - 1)  # Clamp to last segment
         local_t = t - seg_edges[seg_idx]
         row = np.zeros(5) # [x,y,z,time,yaw]
         for j, axis in enumerate(["x","y","z","t","yaw"]):
            if(j==3): row[j] = t
            else:
               c = coeffs[axis][seg_idx]
               row[j] = np.polyval(c[::-1], local_t)
         traj[i] = row
      return traj
      
if __name__  == "__main__":
   import time
   solver = QPSolver2()
   waypoints = np.array([
      [0, 0, 1, 0],
      [2, 2, 3, 90],
      [4, -2, 5, 180],
      [6, 0, 3, -90],
      [8, 0, 1, 0]
   ])
   s = time.time()
   traj = solver.solve(None, waypoints, optimize=True, max_iter=20, desired_velocity=2.0)
   print("Solve time: ", time.time() - s)
   # print(traj)
   # print("Trajectory shape: ", traj.shape)
   profile = solver.profile(traj)
   solver.visualize(traj, waypoints, profile)