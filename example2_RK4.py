import numpy as np
from matplotlib import pyplot
import math

t = 500
n = 5000
dt = t/n

A_T = 0.067
K_R = 0.067
# K_d = 10**(-4)

def KF3(M, C, R):
    dx = [0, 0, 0]
    dx[0] = (A_T - R - K_d + math.sqrt((A_T - R - K_d)**2 + 4*A_T*K_d))/(2*A_T) - M # M
    dx[1] = M - C # C
    dx[2] = C - R/(R/K_R + 1) # R
    return dx

Mlist = []

for K_d in [10**(-3), 10**(-3.5), 10**(-4)]:
    x = [np.zeros(n), np.zeros(n), np.zeros(n)]
    x[0][0] = 0
    x[1][0] = 0
    x[2][0] = 0

    for i in range(1, n):
        M = x[0][i - 1]
        C = x[1][i - 1]
        R = x[2][i - 1]

        k1 = KF3(M, C, R)
        k1_M = k1[0]
        k1_C = k1[1]
        k1_R = k1[2]

        k2 = KF3(M + k1_M * dt/2, C + k1_C * dt/2, R + k1_R * dt/2)
        k2_M = k2[0]
        k2_C = k2[1]
        k2_R = k2[2]

        k3 = KF3(M + k2_M * dt/2, C + k2_C * dt/2, R + k2_R * dt/2)
        k3_M = k3[0]
        k3_C = k3[1]
        k3_R = k3[2]

        k4 = KF3(M + k3_M * dt, C + k3_C * dt, R + k3_R * dt)
        k4_M = k4[0]
        k4_C = k4[1]
        k4_R = k4[2]

        x[0][i] = x[0][i - 1] + dt/6 * (k1_M + 2 * k2_M + 2 * k3_M + k4_M) # M
        x[1][i] = x[1][i - 1] + dt/6 * (k1_C + 2 * k2_C + 2 * k3_C + k4_C) # C
        x[2][i] = x[2][i - 1] + dt/6 * (k1_R + 2 * k2_R + 2 * k3_R + k4_R) # R
    
    Mlist.append(x[0])

M = x[0]
C = x[1]
R = x[2]
ts = np.linspace(0, t, n)

pyplot.plot(ts, Mlist[0])
pyplot.plot(ts, Mlist[1])
pyplot.plot(ts, Mlist[2])
pyplot.grid(True, linestyle='--')
pyplot.legend(('$10^{-3}$', '$10^{-3.5}$', '$10^{-4}$'))
pyplot.show()