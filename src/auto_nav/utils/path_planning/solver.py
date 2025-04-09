import os
import time as ti
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
      if not np.array_equal(waypoints[0],self.current_position) and self.current_position is not None:
         if waypoints.shape[1] == 4: pose = np.append(self.current_position, self.current_orientation)
         else: pose = self.current_position
         waypoints = np.insert(waypoints, 0, pose, axis=0)
      x_points = waypoints[:, 0]
      y_points = waypoints[:, 1]
      z_points = waypoints[:, 2]
      if waypoints.shape[1] == 4:
         if self.current_orientation is not None:
            waypoints[0, 3] = self.current_orientation
         yaw_points = waypoints[:, 3]
         return x_points, y_points, z_points, yaw_points
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
         points = self._parse_waypoints(waypoints)
         if(len(points) == 4):
            x_points, y_points, z_points, yaw_points = points
         else:
            x_points, y_points, z_points = points
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
      points = self._parse_waypoints(self.waypoints)
      yaw_points = None
      if(len(points) == 4):
         x_points, y_points, z_points, yaw_points = points
      else:
         x_points, y_points, z_points = points
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
      psi = cp.Variable(T)
      psid = cp.Variable(T)
      psidd = cp.Variable(T)
      #--- Define initial conditions ---#
      x0, y0, z0 = x_points[0], y_points[0], z_points[0]
      constraints = [X[:, 0] == np.array([x0, y0, z0])]
      constraints += [V[:, 0] == self.current_velocity]
      if yaw_points is not None:
        constraints += [psi[0] == yaw_points[0] * (np.pi/180)]
      #--- Add motion constraints ---#
      for t in range(T - 1):
         constraints += [
            X[:, t+1] == X[:, t] + V[:, t] * dt + 0.5 * A[:, t] * dt**2 + (1/6) * J[:, t] * dt**3 + (1/24) * S[:, t] * dt**4,
            V[:, t+1] == V[:, t] + A[:, t] * dt + 0.5 * J[:, t] * dt**2 + (1/6) * S[:, t] * dt**3,
            A[:, t+1] == A[:, t] + J[:, t] * dt + 0.5 * S[:, t] * dt**2,
            J[:, t+1] == J[:, t] + S[:, t] * dt,
            psi[t+1] == psi[t] + psid[t] * dt + 0.5 * psidd[t] * dt**2,
            psid[t+1] == psid[t] + psidd[t] * dt
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
         #--- Handle yaw constraints ---#
         # if yaw_points is not None: #TODO: Figure this shit out
         #       if yaw_points[i] != -1:
         #          target_yaw = yaw_points[i] * (np.pi/180)
         #          psi_t_idx = psi[int(t_idx)]
         #          angle_diff = psi_t_idx - target_yaw
         #          angle_diff = ca.fmod(psi[int(t_idx)] - target_yaw + ca.pi, 2*ca.pi) - ca.pi
         #          constraints += [ca.fabs(angle_diff) <= yaw_tolerance]
      #--- Define cost function (acceleration, jerk, snap) ---#
      cost = cp.sum_squares(A)
      cost += cp.sum_squares(J)
      cost += cp.sum_squares(S)*10 # penalize snap more
      # cost += cp.sum_squares(psid)
      # cost += cp.sum_squares(psidd)
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
      yaw = psi.value.T
      trajectory = np.insert(trajectory, 3, np.linspace(0, T*dt, T), axis=1)
      trajectory = np.insert(trajectory, 4, yaw, axis=1)
      return trajectory

class CasSolver(BaseSolver):
   def __init__(self):
      super().__init__()
   
   def _solve(self, **kwargs) -> np.ndarray:
      '''
      Formats the problem into a constraint problem and solves it using ipopt.
      Minimizes snap and attempts yaw.
      '''
      points = self._parse_waypoints(self.waypoints)
      yaw_points = None
      if(len(points) == 4):
         x_points, y_points, z_points, yaw_points = points
      else:
         x_points, y_points, z_points = points
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


      
if __name__  == "__main__":
   # waypoints = np.array([  [0, 0, 0], 
   #                         [1, 2, 0],
   #                         [2, 0, 2], 
   #                         [3, -2.2, 2], 
   #                         [1.5, 0, 2], 
   #                         [5, 1, 1],
   #                         [6, 0, 0], 
   #                         [7, 2, 0],
   #                         [8, 0, 0],
   #                         [9, 0, 0]
   #                         ])
   # waypoints = np.array([  [0, 0, 0, 90], 
   #                         [1, 2, 0, 180],
   #                         [2, 0, 2, -1], 
   #                         [3, -2.2, 2, -1], 
   #                         [1.5, 0, 2, -1], 
   #                         [5, 1, 1, -1],
   #                         [6, 0, 0, -1], 
   #                         [7, 2, 0, -1],
   #                         [8, 0, 0, -1],
   #                         [9, 0, 0, -1]
   #                         ])
   # waypoints = np.array([[  1.21      ,  10.24      ,   1.35      ,  -1      ],
   #     [  2.92606891,  12.44012771,   1.35      ,  43.62      ],
   #     [  4.37393109,  13.81987229,   1.35      , 345      ],
   #     [  7.74597775,  11.99999738,   1.35      , 345      ],
   #     [ 12.57402225,  10.70000262,   1.35      ,  53.82      ],
   #     [ 14.11419013,  12.93208394,   1.35      ,  53.82      ],
   #     [ 16.47548592,  16.16074963,   1.35      ,  53.82      ],
   #     [ 16.47548592,  16.16074963,   1.45      ,  90.        ],
   #     [ 17.25158047,  16.0631713 ,   1.75      ,  85.98      ],
   #     [ 17.73321315,  15.47721697,   2.05      ,  81.96      ],
   #     [ 17.94227549,  14.69529186,   2.35      ,  77.94      ],
   #     [ 17.83353107,  13.84701844,   2.65      ,  73.92      ],
   #     [ 17.40271384,  13.06103983,   2.95      ,  69.9       ],
   #     [ 16.69042814,  12.45735231,   3.25      ,  65.88      ],
   #     [ 15.77806617,  12.133396  ,   3.55      ,  61.86      ],
   #     [ 14.77793725,  12.15121548,   3.85      ,  57.84      ],
   #     [ 13.81902815,  12.52850073,   4.15      ,  53.82      ],
   #     [ 14.11419013,  12.93208394,   4.05      ,  53.82      ],
   #     [ 16.18032395,  15.75716642,   4.05      ,  53.82      ],
   #     [ 18.18032395,  15.75716642,   4.05      ,  90.        ],
   #     [ 18.77499031,  15.88005632,   4.05      , 180.        ],
   #     [ 18.80500969,  19.87994368,   4.05      , 250.        ],
   #     [ 18.80500969,  19.87994368,   1.35      , 270.        ],
   #     [ 18.78249515,  16.88002816,   1.35      , 250.        ],
   #     [ 16.7706479 ,  16.56433285,   1.35      , 233.82      ],
   #     [ 14.4093521 ,  13.33566715,   1.35      , 233.82      ],
   #     [ 14.81902815,  12.52850073,   1.35      , 180.        ],
   #     [ 16.04999235,  11.98547524,   1.45      , 180.        ],
   #     [ 16.80048596,  12.20805355,   1.75      , 128.2718142 ],
   #     [ 17.45686885,  12.62974244,   2.05      , 116.31513304],
   #     [ 17.95170916,  13.22275585,   2.35      , 108.07769067],
   #     [ 18.22362757,  13.92954066,   2.65      , 101.59473183],
   #     [ 18.23162321,  14.66396083,   2.95      ,  96.16637445],
   #     [ 17.96893962,  15.31912966,   3.25      ,  91.45325067],
   #     [ 17.473598  ,  15.78155123,   3.55      ,  87.26222772],
   #     [ 16.83295695,  15.9500776 ,   3.85      ,  83.47179212],
   #     [ 16.18032395,  15.75716642,   4.15      ,  80.        ],
   #     [ 16.18032395,  15.75716642,   4.05      ,  53.82      ],
   #     [ 14.4093521 ,  13.33566715,   4.05      ,  53.82      ],
   #     [ 12.0912178 ,  10.83000209,   1.35      ,  53.82      ],
   #     [  9.1943911 ,  11.60999895,   1.35      ,  43.82      ],
   #     [  8.2287822 ,  10.86999791,   1.35      ,  20.        ],
   #     [  1.84017226,   9.40531929,   1.35      ,  43.62      ],
   #     [  1.21      ,  10.24      ,   1.        ,  43.62      ]])
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
   waypoints = np.array([
      [20,10,1.45, 0],
      [34,12,1.45, 0],
      [20,16,1.45, 0],
      [12, 14, 1.45, 90],
      [20,10,1.45, 90]
   ])
   solver = QPSolver()
   # solver.set_hard_constraints(max_tolerance=0.2)
   # solver.set_hard_constraints(max_velocity=2, max_acceleration=5, max_tolerance=0.2)
   trajectory = solver.solve([18,10,0], waypoints, current_orientation=0)
   # for i in trajectory:
   #    print("Line:", i)
   profile = solver.profile(trajectory)
   solver.visualize(trajectory, waypoints, profile)

   # solver.set_hard_constraints(max_jerk=3)
   # solver.temporal_scale(trajectory)
   # profile = solver.profile(trajectory)
   # solver.visualize(trajectory, waypoints, profile)