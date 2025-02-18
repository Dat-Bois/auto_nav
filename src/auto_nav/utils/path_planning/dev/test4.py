# 1D Reference Trajectory Optimization with Position, Velocity, Acceleration, and Jerk Constraints
# Used GPT

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Step 1: Define problem parameters
T = 50         # Number of timesteps
dt = 0.1       # Time step size (seconds)
v_max = 3.0    # Maximum velocity (m/s)
a_max = 2.0    # Maximum acceleration (m/s²)
j_max = None   # No jerk constraint

# Step 2: Define optimization variables
x = cp.Variable(T)  # Position
v = cp.Variable(T)  # Velocity
a = cp.Variable(T)  # Acceleration
j = cp.Variable(T)  # Jerk (unconstrained)

# Step 3: Define initial conditions
x0, v0, a0 = 0.0, 0.0, 0.0
constraints = [x[0] == x0, v[0] == v0, a[0] == a0]

# Step 4: Add motion constraints
for t in range(T - 1):
    constraints += [
        x[t+1] == x[t] + v[t] * dt + 0.5 * a[t] * dt**2 + (1/6) * j[t] * dt**3, # pos = pos + v*dt + 0.5*a*dt^2 + (1/6)*j*dt^3
        v[t+1] == v[t] + a[t] * dt + 0.5 * j[t] * dt**2,
        a[t+1] == a[t] + j[t] * dt
    ]

# Velocity and acceleration limits
constraints += [cp.abs(v) <= v_max]
constraints += [cp.abs(a) <= a_max]

# Step 5: Define waypoints and tolerance
waypoints = [0.0, 4.0, 3.0]  # Positions
waypoint_times = [0, 25, 49]  # Time indices for waypoints
tolerance = 0.2  # Allow ±0.2 meters deviation

for i, t_idx in enumerate(waypoint_times):
    constraints.append(x[t_idx] >= waypoints[i] - tolerance)
    constraints.append(x[t_idx] <= waypoints[i] + tolerance)

# Step 6: Define cost function (velocity tracking, acceleration, jerk)
v_desired = np.ones(T) * 3.0  # Target velocity of 1 m/s
# cost = cp.sum_squares(v - v_desired)  # Minimize velocity tracking error
cost = cp.sum_squares(a)  # Minimize acceleration effort
cost += cp.sum_squares(j)  # Minimize jerk effort

# Step 7: Solve the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve(solver=cp.OSQP)

# Step 8: Extract optimized trajectory
x_opt, v_opt, a_opt, j_opt = x.value, v.value, a.value, j.value
time = np.arange(T) * dt

# Step 9: Plot results
plt.figure(figsize=(10, 6))
plt.subplot(3,1,1)
plt.plot(time, x_opt, label="Position (m)")
plt.scatter(np.array(waypoint_times) * dt, waypoints, color='red', label="Waypoints")
plt.ylabel("Position")
plt.legend()

plt.subplot(3,1,2)
plt.plot(time, v_opt, label="Velocity (m/s)")
plt.plot(time, v_desired, "--", label="Desired Velocity", alpha=0.7)
plt.ylabel("Velocity")
plt.legend()

plt.subplot(3,1,3)
plt.plot(time, a_opt, label="Acceleration (m/s²)")
plt.ylabel("Acceleration")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.show()
