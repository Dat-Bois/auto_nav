import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

# https://cookierobotics.com/078/

def get_Q(T):
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1152*T, 2880*T**2, 5760*T**3, 10080*T**4],
        [0, 0, 0, 0, 2880*T**2, 9600*T**3, 21600*T**4, 40320*T**5],
        [0, 0, 0, 0, 5760*T**3, 21600*T**4, 51840*T**5, 100800*T**6],
        [0, 0, 0, 0, 10080*T**4, 40320*T**5, 100800*T**6, 201600*T**7]
    ])

def get_coefficients(T, wp):
    N = 8 # Septic polynomial has 8 coefficients
    M = len(T) # Number of polynomial segments
    L = 14 # Number of constraints

    # Q Matrix
    Z = np.zeros((N, N))
    Q0 = get_Q(T[0])
    Q1 = get_Q(T[1])
    Q2 = get_Q(T[2])
    Q = np.block([
        [Q0, Z, Z],
        [Z, Q1, Z],
        [Z, Z, Q2]
    ])

    f = np.zeros(N*M)

    # Constraints (Ap=b)
    A = np.zeros((L, N*M))
    # Start position of polynomials
    A[0,:8] = [1, 0, 0, 0, 0, 0, 0, 0]
    A[1,8:16] = [1, 0, 0, 0, 0, 0, 0, 0]
    A[2,16:] = [1, 0, 0, 0, 0, 0, 0, 0]
    # End position of polynomials
    A[3,:8] = [1, T[0], T[0]**2, T[0]**3, T[0]**4, T[0]**5, T[0]**6, T[0]**7]
    A[4,8:16] = [1, T[1], T[1]**2, T[1]**3, T[1]**4, T[1]**5, T[1]**6, T[1]**7]
    A[5,16:] = [1, T[2], T[2]**2, T[2]**3, T[2]**4, T[2]**5, T[2]**6, T[2]**7]
    # Continuous velocity
    A[6,:16] = [0, 1, 2*T[0], 3*T[0]**2, 4*T[0]**3, 5*T[0]**4, 6*T[0]**5, 7*T[0]**6, 0, -1, 0, 0, 0, 0, 0, 0]
    A[7,8:] = [0, 1, 2*T[1], 3*T[1]**2, 4*T[1]**3, 5*T[1]**4, 6*T[1]**5, 7*T[1]**6, 0, -1, 0, 0, 0, 0, 0, 0]
    # Continuous acceleration
    A[8,:16] = [0, 0, 2, 6*T[0], 12*T[0]**2, 20*T[0]**3, 30*T[0]**4, 42*T[0]**5, 0, 0, -2, 0, 0, 0, 0, 0]
    A[9,8:] = [0, 0, 2, 6*T[1], 12*T[1]**2, 20*T[1]**3, 30*T[1]**4, 42*T[1]**5, 0, 0, -2, 0, 0, 0, 0, 0]
    # Continuous jerk
    A[10,:16] = [0, 0, 0, 6, 24*T[0], 60*T[0]**2, 120*T[0]**3, 210*T[0]**4, 0, 0, 0, -6, 0, 0, 0, 0]
    A[11,8:] = [0, 0, 0, 6, 24*T[1], 60*T[1]**2, 120*T[1]**3, 210*T[1]**4, 0, 0, 0, -6, 0, 0, 0, 0]
    # Continuous snap
    A[12,:16] = [0, 0, 0, 0, 24, 120*T[0], 360*T[0]**2, 840*T[0]**3, 0, 0, 0, 0, -24, 0, 0, 0]
    A[13,8:] = [0, 0, 0, 0, 24, 120*T[1], 360*T[1]**2, 840*T[1]**3, 0, 0, 0, 0, -24, 0, 0, 0]

    b = np.zeros(L)
    b[:6] = [wp[0], wp[1], wp[2], wp[1], wp[2], wp[3]]

    sol = solvers.qp(matrix(Q), matrix(f), None, None, matrix(A), matrix(b))
    return list(sol['x'])

# Waypoints
wp_t = np.array([0., 10., 30., 32.])
wp_x = np.array([0., 5., 10., 3.])
T = np.ediff1d(wp_t)

p = get_coefficients(T, wp_x)

N = 500
t = np.linspace(wp_t[0], wp_t[-1], N)
pos = [None] * N
vel = [None] * N
acc = [None] * N
jrk = [None] * N
snp = [None] * N
for i in range(N):
    j = np.nonzero(t[i] <= wp_t)[0][0] - 1
    j = np.max([j, 0])
    ti = t[i] - wp_t[j]
    x_coeff = np.flip(p[8*j:8*j+8])
    v_coeff = np.polyder(x_coeff)
    a_coeff = np.polyder(v_coeff)
    j_coeff = np.polyder(a_coeff)
    s_coeff = np.polyder(j_coeff)
    pos[i] = np.polyval(x_coeff, ti)
    vel[i] = np.polyval(v_coeff, ti)
    acc[i] = np.polyval(a_coeff, ti)
    jrk[i] = np.polyval(j_coeff, ti)
    snp[i] = np.polyval(s_coeff, ti)

plt.subplot(5, 1, 1)
plt.plot(t, pos, label='Position')
plt.plot(wp_t, wp_x, '.', label='Waypoint')
plt.legend()
plt.subplot(5, 1, 2)
plt.plot(t, vel, label='velocity')
plt.legend()
plt.subplot(5, 1, 3)
plt.plot(t, acc, label='acceleration')
plt.legend()
plt.subplot(5, 1, 4)
plt.plot(t, jrk, label='jerk')
plt.legend()
plt.subplot(5, 1, 5)
plt.plot(t, snp, label='snap')
plt.legend()
plt.xlabel('Time')
# plt.show()
plt.savefig('test2.png')