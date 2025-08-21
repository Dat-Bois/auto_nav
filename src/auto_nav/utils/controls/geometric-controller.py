import numpy as np

# Basic test implementation

# Rotor constants
# Force = Fk * (angular_velocity)^2 (measured in Newtons)
Fk = 0.1
# Moment = Mk * (angular_velocity)^2 (measured in Newton-meters)
Mk = 0.1

# Drone characteristics
M = 1.0  # Mass of the drone in kg
# Assuming a perfect distanced quadrotor with 4 rotors
L = 0.5  # Distance from the center of the quadrotor to each rotor in meters
G = 9.81  # Acceleration due to gravity in m/s^2

# Motor Matrix
MOTOR_MIX = np.array([
        [Fk, Fk, Fk, Fk], #u1
        [0, Fk*L, 0, -Fk*L], #u2
        [-Fk*L, 0, Fk*L, 0], #u3
        [Mk, -Mk, Mk, -Mk]  #u4
    ])

E3 = np.array([0,0,1]).T

# TEST VARIABLES

# Desired state (world frame)
# This can come from either the trajectory or solved via dt
desired_pstate = np.array([2, 0, 2])  # [x, y, z]
desired_vstate = np.array([1, 0, 0])  # [vx, vy, vz]
desired_astate = np.array([0.3, 0, 0])  # [ax, ay, az]
desired_jstate = np.array([0.1, 0, 0])  # [ax, ay, az]
desired_psi = 1.2  # Desired yaw angle in radians
desired_vpsi = 0.02  # Desired yaw rate in radians
# Current state (world frame) 
# This can be measured or derived from the position estimate and derived
current_state = np.array([0, 0, 2])  # [x, y, z]
current_vstate = np.array([0, 0, 0])  # [vx, vy, vz]
current_astate = np.array([0, 0, 0])  # [ax, ay, az]
current_jstate = np.array([0, 0, 0])  # [ax, ay, az]
current_psi = 1.1  # Current yaw angle in radians
current_pqr = np.array([0, 0, 0])  # Current angular velocity in body frame [p, q, r]

# Controller

eP = desired_pstate - current_state  # Position error
eV = desired_vstate - current_vstate  # Velocity error

Kp = np.identity(3) * 1.0  # Proportional gain for position control
Kv = np.identity(3) * 0.5  # Derivative gain for velocity control

# World frame
Fdes = -(Kp @ eP.T + Kv @ eV.T) + np.array([0, 0, M * G]).T + M*current_astate

# Calculate the desired rotation matrix from world frame to body frame
XcDes = np.array([np.cos(desired_psi), np.sin(desired_psi), 0])  # Align x-axis with desired yaw
# Unit vector for desired z-axis in world frame
ZbDes = Fdes / np.linalg.norm(Fdes)
# Then we can take the cross product of desired Zb and Xc which gives us the orthogonal vector to both Zb and Xc that is Yb.
YbDes = np.cross(ZbDes, XcDes)
YbDes /= np.linalg.norm(YbDes)  # Normalize YbDes
# Orthagonal vector to both Zb and YbDes gives us XbDes (both are already normalized (unit vectors))
XbDes = np.cross(YbDes, ZbDes)

# Do the same but for the current state
Xc = np.array([np.cos(current_psi), np.sin(current_psi), 0])
t = current_astate + np.array([0, 0, G]) 
Zb = t / np.linalg.norm(t)
Yb = np.cross(Zb, Xc)
Yb /= np.linalg.norm(Yb)  # Normalize Yb
Xb = np.cross(Yb, Zb)
Rcurr = np.column_stack((Xb, Yb, Zb))  # Current rotation matrix from world frame to body frame

# Check for XcDes and ZbDes being parallel (or close to it)
if(np.dot(Xb, XbDes)<0): 
    # If they are parallel, we need to adjust the desired XbDes to avoid singularity
    XbDes = -XbDes
    YbDes = -YbDes 
# Create the desired rotation matrix from world frame to body frame
Rdes = np.column_stack((XbDes, YbDes, ZbDes))

# Now we can calculate the desired thrust vector in body frame
u1 = Fdes @ Zb  # Total thrust required (scalar) in the direction of Zb

Rerr = 1/2 * (Rdes.T @ Rcurr - Rcurr.T @ Rdes)  # Skew-symmetric matrix of the rotation error
Rerr = np.array([[Rerr[2, 1]], [Rerr[0, 2]], [Rerr[1, 0]]])  # Vee mapping to vector form
print("Rotation Error (Rerr):", Rerr)

# Current angular velocity is measured in body frame (p, q, r from gryo).
# Desired is computed via the trajectory
# The end goal is to get the angular velocity in body frame.

# Desired
HwDes = M/u1 * (desired_jstate-(ZbDes @ desired_jstate)*ZbDes)
pDes = -HwDes @ YbDes
qDes = HwDes @ XbDes
rDes = desired_vpsi*E3 @ ZbDes

Wdes = pDes*XbDes + qDes*YbDes + rDes*ZbDes  # Desired angular velocity in body frame
Wcurr = current_pqr

# Calculate the error in angular velocity
eW = (Wcurr - Wdes)[np.newaxis]  # Angular velocity error
print("Angular Velocity Error:", eW)

# Diagonal gain matrix for angular velocity control
Kr = np.identity(3) * 0.1 # P gain
Kw = np.identity(3) * 0.05 # D gain
controls = (-Kr @ Rerr - Kw @ eW.T).T

controls = np.insert(controls, 0, u1, axis=1)
print("Controls:")
print("U1 (Thrust):", u1)
print("U2 (Roll Moment):", controls[0, 1])
print("U3 (Pitch Moment):", controls[0, 2])
print("U4 (Yaw Moment):", controls[0, 3])

# Rotor speeds
rotor_speeds = np.linalg.pinv(MOTOR_MIX) @ controls.T
print("Rotor Speeds:", rotor_speeds)