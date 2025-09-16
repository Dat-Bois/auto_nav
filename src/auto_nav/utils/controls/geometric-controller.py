import numpy as np
from auto_nav import Profile

class QuadProperties():
    def __init__(self, mass: float, arm_length: float, Fk: float, Mk: float):
        self.M = mass
        self.L = arm_length
        self.Fk = Fk
        self.Mk = Mk

class DState():
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.accel = np.zeros(3)
        self.jerk = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0

class CState():
    def __init__(self):
        pass
    def from_drake(self, state: np.ndarray):
        '''
        state: [x,y,z,r,p,y,vx,vy,vz,rd,pd,yd,ax,ay,az,rdd,pdd,ydd] (18-dim vector)
        '''
        self.pos = state[0:3] # required
        self.rot = state[3:6] # required
        self.vel = state[6:9] # required
        self.omega = state[9:12] # required
        self.accel = state[12:15] # required
        self.alpha = state[15:18]
        return self

class GeometricController():
    def __init__(self, quad_props: QuadProperties):
        self.qps = quad_props
        self.G = -9.81
        self.E3 = np.array([0,0,1])
        # Motor 1: +X # Motor 2: +Y # Motor 3: -X # Motor 4: -Y
        self.MOTOR_MIX = np.array([
        [quad_props.Fk, quad_props.Fk, quad_props.Fk, quad_props.Fk], #u1
        [0, quad_props.Fk*L, 0, -quad_props.Fk*L], #u2
        [-quad_props.Fk*L, 0, quad_props.Fk*L, 0], #u3
        [quad_props.Mk, -quad_props.Mk, quad_props.Mk, -quad_props.Mk]])  #u4

        # Adjusted externally
        self.Kp = np.identity(3) * 1.0  # P gain for position control
        self.Kv = np.identity(3) * 0.5  # D gain for velocity control
        # Diagonal gain matrix for angular velocity control
        self.Kr = np.identity(3) * 0.1 # P gain
        self.Kw = np.identity(3) * 0.05 # D gain

    def compute_control(self, dstate: DState, cstate: CState):
        '''
        Reference the playground jupiter notebook for detailed derivation and explanation.
        Returns the desired rotor speeds to achieve the desired state.
        '''
        G = -9.81
        q = self.qps
        eP = dstate.pos - cstate.pos  # Position error
        eV = dstate.vel - cstate.vel  # Velocity error
        Fdes = -(self.Kp @ eP.T + self.Kv @ eV.T) + np.array([0, 0, q.M * G]).T + q.M*dstate.accel
        XcDes = np.array([np.cos(dstate.yaw), np.sin(dstate.yaw), 0])
        ZbDes = Fdes / np.linalg.norm(Fdes)
        YbDes = np.cross(ZbDes, XcDes)
        YbDes /= np.linalg.norm(YbDes)
        XbDes = np.cross(YbDes, ZbDes)
        # Current state
        Xc = np.array([np.cos(cstate.rot[2]), np.sin(cstate.rot[2]), 0])
        t = cstate.accel + np.array([0, 0, G]) 
        Zb = t / np.linalg.norm(t)
        Yb = np.cross(Zb, Xc)
        Yb /= np.linalg.norm(Yb)
        Xb = np.cross(Yb, Zb)
        Rcurr = np.column_stack((Xb, Yb, Zb)) 
        if(np.dot(Xb, XbDes)<0): 
            XbDes = -XbDes
            YbDes = -YbDes 
        Rdes = np.column_stack((XbDes, YbDes, ZbDes))
        u1 = Fdes @ Zb
        Rerr = 1/2 * (Rdes.T @ Rcurr - Rcurr.T @ Rdes)
        Rerr = np.array([[Rerr[2, 1]], [Rerr[0, 2]], [Rerr[1, 0]]])
        HwDes = q.M/u1 * (dstate.jerk-(ZbDes @ dstate.jerk)*ZbDes)
        pDes = -HwDes @ YbDes
        qDes = HwDes @ XbDes
        rDes = dstate.yaw_rate*self.E3 @ ZbDes
        Wdes = pDes*XbDes + qDes*YbDes + rDes*ZbDes
        Wcurr = cstate.omega
        eW = (Wcurr - Wdes)[np.newaxis]
        controls = (-self.Kr @ Rerr - self.Kw @ eW.T).T
        controls = np.insert(controls, 0, u1, axis=1)
        rotor_speeds = np.linalg.pinv(self.MOTOR_MIX) @ controls.T
        return np.clip(rotor_speeds, 0, None).flatten()
