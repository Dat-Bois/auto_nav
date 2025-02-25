'''
This file opens up an API to the path planning algorithms.
Given a set of waypoints, (which can be added/adjusted/removed) replan trajectory
Use MPC to provide a local horizon trajectory
Able to determine when a waypoint has been passed based on current location
Able to return next goto position
'''

import numpy as np
from .solver import BaseSolver

class Planner:

    def __init__(self, waypoints : np.ndarray, solver : BaseSolver):
        self.waypoints = waypoints
        self.solver = solver

        self.current_position = None
        self.current_velocity = None
        
        self.traj = None
        self.remaining_waypoints = waypoints

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

    def update_state(self, position : np.ndarray, velocity : np.ndarray | None = None):
        '''
        Update the current state of the vehicle.
        '''
        assert len(position) == 3
        if velocity is not None:
            assert len(velocity) == 3
        self.current_position = position
        self.current_velocity = velocity

    def plan_global(self, waypoints : np.ndarray | None = None, max_time: float | None = None):
        '''
        Plan a global trajectory. Max time is the maximum time the trajectory can take.
        Returns a trajectory.
        '''
        if waypoints is None:
            waypoints = self.waypoints
        if waypoints is None:
            raise ValueError('No waypoints provided')
        traj = self.solver.solve(waypoints, self.current_position, self.current_velocity)
        if traj is None:
            # Relax constraints and replan, but save the original constraints and then reapply them for temporal scaling
            constraints = self.solver.get_hard_constraints()
            self.solver.set_hard_constraints({})
            traj = self.solver.solve(waypoints, self.current_position, self.current_velocity)
            self.solver.set_hard_constraints(constraints)
        return self.solver.temporal_scale(traj, max_time)
        

