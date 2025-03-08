import numpy as np
from scipy.spatial.transform import Rotation

gates = np.array([ # xyz, yaw
    [3.65, 13.13, 1.45, -46.38], # 0
    [10.16, 11.35, 1.45, -105.07], # 1
    [15.59, 14.95, 1.45, -36.18], # 2
    [15.59, 14.95, 4.15, -36.18], # 3
    [18.79, 17.88, 4.15, -0.43], # 4
    [18.79, 17.88, 1.45, -0.43], # 5
    [20.51, 9.9, 1.45, 0], # 6
    [15.75, 1.83, 1.45, -90], # 7
    [10.25, 5.35, 1.45, -90], # 8
    [4.75, 1.83, 1.45,-90] # 9
])
gates[:, 3] += 90 # undo change of coordinates that was necessary in gazebo

def wp_inline_with_gate(gate, distance, invert_yaw = False, elevation_offset = -0.1):
    '''invert_yaw affects only the yaw of the waypoint, not the position'''
    gate_pos = gate[:3]
    gate_yaw = gate[3]
    direction_vector = Rotation.from_euler('z', gate_yaw, degrees=True).apply([1, 0, 0])
    res = np.array([
        *(gate_pos + distance*direction_vector),
        gate_yaw + 180*int(invert_yaw)
    ])
    res[2]+=elevation_offset
    return res

def spiral_position(t, start_yaw, total_yaw, start_elev, end_elev, radius_start, radius_end, spiral_bend_fn):
    return Rotation.from_euler('z', start_yaw + t*total_yaw, degrees=True).apply(np.array([radius_start+spiral_bend_fn(t)*(radius_end-radius_start),0,0])) + np.array([0,0,start_elev + t*(end_elev-start_elev)])


path = np.array([ # xyz, yaw
    (1.21, 10.24, 1.35, 43.62), # takeoff point
    wp_inline_with_gate(gates[0], -1), # gate 1
    (*wp_inline_with_gate(gates[0], 1)[:3], gates[1][3]), # gate 1
    wp_inline_with_gate(gates[1], -2.5), # gate 2
    (*wp_inline_with_gate(gates[1], 2.5)[:3], gates[2][3]), # gate 2
    wp_inline_with_gate(gates[2], -2.5), # gate 3 bottom
    wp_inline_with_gate(gates[2], 1.5), # gate 3 bottom
    *[
        (*(gates[2][:3] + spiral_position(t, gates[2][3], -180, 0, 2.7, 1.5, 3, spiral_bend_fn=np.sqrt)), 90 + t*(gates[3][3] - 90))
        for t in np.linspace(0,1,10)
    ],
    wp_inline_with_gate(gates[3], -2.5), # gate 3 top
    wp_inline_with_gate(gates[3], 1), # gate 3 top
    (*(wp_inline_with_gate(gates[3], 1)[:3] + np.array([2, 0, 0])), 90), # reverse maneuver start
    (*wp_inline_with_gate(gates[4], -2.0)[:3], 180), # right before entering split-S
    (*wp_inline_with_gate(gates[4], 2.0)[:3], 250), # top of split S in the corner
    (*wp_inline_with_gate(gates[5], 2.0)[:3], 270), # bottom of split S in the corner
    (*wp_inline_with_gate(gates[5], -1.0)[:3], 250), # out the bottom of split-S
    wp_inline_with_gate(gates[2], 2.0, True),
    wp_inline_with_gate(gates[2], -2.0, True),
    (*(wp_inline_with_gate(gates[2], -3.0)[:3]+np.array([1,0,0])), 180),
    *[
        (*(gates[2][:3] + spiral_position(t, 180+gates[2][3]+45, 135, 0, 2.7, 3, 1, spiral_bend_fn=lambda x: x**3)), 180+(t**0.3)*(-100))
        for t in np.linspace(0,1,10)
    ],
    (*wp_inline_with_gate(gates[3], 1.0)[:3], gates[3][3]),
    (*wp_inline_with_gate(gates[3], -2.0)[:3], gates[3][3]),
    (*wp_inline_with_gate(gates[1], 2.0)[:3], gates[3][3]),
    (*wp_inline_with_gate(gates[1], -1.0)[:3], gates[3][3]-10),
    (*(wp_inline_with_gate(gates[1], -2.0)[:3]+np.array([0,-1,0])), 20),
    wp_inline_with_gate(gates[0], -2.5) + np.array([0,-2,0,0]),
    (1.21, 10.24, 1, 43.62), # takeoff point
])

print(repr(path))