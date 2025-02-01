import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x_points = np.array([0, 1, 2, 3, 1.5, 5])
y_points = np.array([0, 2, 0, -2, 0, 1])
z_points = np.array([0, 0, 2, 2, 2, 1])

t = np.linspace(0, 1, len(x_points))  # Parameter (uniform)
# print(t)
# t = np.array([0, 0.2, 0.3, 0.5, 0.8, 1])

arc_length = np.cumsum(np.sqrt(np.diff(x_points)**2 + np.diff(y_points)**2))
arc_length = np.insert(arc_length, 0, 0)
t = arc_length / arc_length[-1]
print(t)
spline_x = CubicSpline(t, x_points, bc_type='natural')
spline_y = CubicSpline(t, y_points, bc_type='natural')
spline_z = CubicSpline(t, z_points, bc_type='natural')

t_fine = np.linspace(0, 1, 100)
x_smooth = spline_x(t_fine)
y_smooth = spline_y(t_fine)
z_smooth = spline_z(t_fine)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x_smooth, y_smooth, z_smooth, label='Smooth Trajectory')
ax.scatter(x_points, y_points, z_points, color='red', label='Waypoints')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
#plt.show()
plt.savefig('test3.png')
