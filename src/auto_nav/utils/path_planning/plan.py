'''
This file opens up an API to the path planning algorithms.
Given a set of waypoints, (which can be added/adjusted/removed) replan trajectory
Use MPC to provide a local horizon trajectory
Able to determine when a waypoint has been passed based on current location
Able to return next goto position
'''

import time
import numpy as np
from .solver import BaseSolver

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

    def set_hard_constraints(self, *, constraints : dict = None, **kwargs):
        '''
        Set hard constraints for the solver.
        '''
        if constraints is not None:
            self.solver.set_hard_constraints(constraints)
        self.solver.set_hard_constraints(kwargs)

    '''
     - Need to implement a method to determine location in trajectory.
        - Based off this determine which waypoints have been passed.
     - Determine distance from current position to intended trajectory location.
        - If distance is too high, replan trajectory from that position using remaining waypoints.
     - Determine next goto position or velocity.
    '''

    def update_state(self, *, position : np.ndarray | None = None, velocity : np.ndarray | None = None, heading : float | None = None):
        '''
        Update the current state of the vehicle.
        TODO: Implement UKF state estimation.
        '''
        if position is not None:
            assert len(position) == 3
            self.current_position = position
        if velocity is not None:
            assert len(velocity) == 3
            self.current_velocity = velocity
        if heading is not None:
            assert isinstance(heading, float)
            self.current_orientation = heading
        if position is not None or velocity is not None or heading is not None:
            self.last_update_time = time.time()

    def _get_next_point(self, traj : np.ndarray, position : np.ndarray, velocity : np.ndarray):
        '''
        Get the closest point on the trajectory to the current position.
        
        Basic method:
        - Use the current position to determine the closest point on the trajectory starting from the front.
         - Use the direction of the current velocity to make a plane and ensure the point is behind the plane.
           - If the point is in front, then we haven't passed the waypoint yet, so return this point.
        - Remove all waypoints before this point in the remaining waypoints.
        - Remove all points before this point in the remaining trajectory.
        - Return the next point on the trajectory.
        '''
        pass

    def get_next_goto(self):
        '''
        Get the next goto position.
        '''
        if self.traj is None:
            return None
        if self.current_position is None or self.current_velocity is None:
            return None
        if self.remaining_trajectory is None:
            return None
       
        # Get the closest point on the trajectory



    def plan_global(self, waypoints : np.ndarray | None = None, max_time: float | None = None):
        '''
        Plan a global trajectory. Max time is the maximum time the trajectory can take.
        Returns a trajectory.
        '''
        if waypoints is None:
            waypoints = self.waypoints
        if waypoints is None:
            raise ValueError('No waypoints provided')
        traj = self.solver.solve(waypoints, self.current_position, self.current_velocity, self.current_orientation)
        if traj is None:
            # Relax constraints and replan, but save the original constraints and then reapply them for temporal scaling
            constraints = self.solver.get_hard_constraints()
            self.solver.set_hard_constraints({})
            traj = self.solver.solve(waypoints, self.current_position, self.current_velocity)
            self.solver.set_hard_constraints(constraints)
            return self.solver.temporal_scale(traj, max_time)
        return traj
        

