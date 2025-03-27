'''
This file opens up an API to the path planning algorithms.
Given a set of waypoints, (which can be added/adjusted/removed) replan trajectory
Use MPC to provide a local horizon trajectory
Able to determine when a waypoint has been passed based on current location
Able to return next goto position
'''


'''
    - Need to implement a method to determine location in trajectory.
    - Based off this determine which waypoints have been passed.
    - Determine distance from current position to intended trajectory location.
    - If distance is too high, replan trajectory from that position using remaining waypoints.
    - Determine next goto position or velocity.
'''

import time
import numpy as np
from .solver import BaseSolver
from auto_nav import DroneState

class Planner:

    def __init__(self, waypoints : np.ndarray, solver : BaseSolver):
        self.waypoints = waypoints
        self.solver = solver

        self.current_position = None
        self.current_velocity = None
        self.current_orientation = None
        self.last_update_time = None
        
        self.traj = None
        self.remaining_waypoints = waypoints
        self.remaining_trajectory = None

        self.v_err = 0
        self.old_time = None

    def set_trajectory(self, traj : np.ndarray):
        '''
        Set a trajectory for the planner.
        '''
        self.traj = traj
        self.remaining_trajectory = traj

    def set_hard_constraints(self, *, constraints : dict = None, **kwargs):
        '''
        Set hard constraints for the solver.
        '''
        if constraints is not None:
            self.solver.set_hard_constraints(kwargs = constraints)
        else:
            self.solver.set_hard_constraints(kwargs = kwargs)

    def update_state(self, *,   position : np.ndarray | None = None, 
                                velocity : np.ndarray | None = None, 
                                heading : float | None = None,
                                state : DroneState | None = None):
        '''
        Update the current state of the vehicle.
        TODO: Implement UKF state estimation.
        '''
        if state is not None:
            position = np.array([state.pos.x, state.pos.y, state.pos.z])
            velocity = np.array([state.lin_vel.x, state.lin_vel.y, state.lin_vel.z, state.ang_vel.z])
            heading = state.euler.yaw
        if position is not None:
            assert len(position) == 3
            self.current_position = position
        if velocity is not None:
            assert len(velocity) == 4
            self.current_velocity = velocity
        if heading is not None:
            assert isinstance(heading, float)
            self.current_orientation = heading
        if position is not None or velocity is not None or heading is not None:
            self.last_update_time = time.time()

    def _find_lookahead(self, *, K : float = 1.5):
        look_dist = K * np.linalg.norm(self.current_velocity[:3])
        dists = np.linalg.norm(self.remaining_trajectory[:, :3] - self.current_position, axis=1)
        idx = np.argmin(dists)
        # Check if the idx is in front or behind the current position
        if np.dot(self.remaining_trajectory[idx, :3] - self.current_position, self.current_velocity[:3]) < 0:
            idx += 1
        if np.all(self.remaining_trajectory[idx,:3] == self.current_position):
            idx += 1
        return idx
        # arc_dists = np.cumsum(np.linalg.norm(np.diff(self.traj[:, :3], axis=0), axis=1))
        # arc_dists = np.insert(arc_dists, 0, 0)
        # for i in range(idx, len(self.traj)):
        #     if arc_dists[i] - arc_dists[idx] >= look_dist:
        #         return self.traj[i]
        # return self.traj[-1]

    def _find_next(self):
        dists = np.linalg.norm(self.remaining_trajectory[:, :3] - self.current_position, axis=1)
        # filter only the dists < 0.3
        for idx, dist in enumerate(dists):
            if dist < 0.3:
               break 
        # ensure idx is in front of current position
        if np.dot(self.remaining_trajectory[idx, :3] - self.current_position, self.current_velocity[:3]) < 0 or np.all(self.remaining_trajectory[idx,:3] == self.current_position):
            idx += 1
        # delete trajectory up to idx
        self.remaining_trajectory = self.remaining_trajectory[idx:]
        return idx
    
    def next_velocity(self, *, K : float = 1.5, KP : float = 1.0, KD : float = 0.1, velocities : np.ndarray | None = None):
        '''
        Determine the next velocity command.
        DOESN'T WORK 
        '''
        if self.current_position is None or self.current_velocity is None or self.current_orientation is None:
            raise ValueError('State not updated')
        if self.traj is None:
            raise ValueError('Trajectory not planned')
        idx = self._find_next()
        if idx is None:
            return np.array([0, 0, 0])
        if idx >= len(velocities):
            return np.array([0, 0, 0])
        v = self.current_velocity[:3]
        v_des = velocities[idx,:3]
        print(v_des)
        return v_des

    def plan_global(self, *, waypoints : np.ndarray | None = None, max_time: float | None = None, set_time: float | None = None):
        '''
        Plan a global trajectory. Max time is the maximum time the trajectory can take.
        Returns a trajectory.
        '''
        if waypoints is None:
            waypoints = self.waypoints
        if waypoints is None:
            raise ValueError('No waypoints provided')
        traj = self.solver.solve(self.current_position, waypoints, self.current_velocity, self.current_orientation)
        if traj is None:
            print('***Trajectory could not be solved, applying temporal scaling...***')
            # Relax constraints and replan, but save the original constraints and then reapply them for temporal scaling
            constraints = self.solver.get_hard_constraints()
            self.set_hard_constraints(constraints = {})
            traj = self.solver.solve(self.current_position, waypoints, self.current_velocity, self.current_orientation)
            self.set_hard_constraints(constraints = constraints)
            if set_time is not None:
                time_var = traj[:, 3]
                multiplier = set_time / time_var[-1]
                time_var = time_var * multiplier
                traj[:, 3] = time_var
                return traj
            return self.solver.temporal_scale(traj, max_time)
        if set_time is not None:
            time_var = traj[:, 3]
            multiplier = set_time / time_var[-1]
            time_var = time_var * multiplier
            traj[:, 3] = time_var
        return traj
        

