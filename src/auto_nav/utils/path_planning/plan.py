'''
What defines a trajectory optimization for a quadcopter?
 - Hit all waypoints in sequence within a certain radius
 - Remove redundant waypoints
 - Horizon lookahead method to adjust trajectory as the vehicle moves
 - Add in spline interpolation between waypoints
    - Advanced: Minimize the 3rd and 4th derivative of the position (acceleration and jerk)
    - Advanced: Attempt to maintain the same velocity throughout the trajectory
'''