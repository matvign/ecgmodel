import tkinter
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def odefcn(T, Y, a, b, w, events):
    x, y, z = Y
    dy = np.zeros(3)

    theta = np.arctan2(y, x)
    alpha = 1.0 - np.sqrt(x**2 + y**2)
    dy[0] = alpha*x - w*y
    dy[1] = alpha*y + w*x

    dy[2] = -(z - 0)
    for i in range(0, 5):
        dtheta = theta - events[i]
        dy[2] -= a[i] * dtheta * np.exp(-(dtheta**2) / (2 * b[i]**2))

    return dy


pi = np.pi
events = np.array([-pi/3, -pi/12, 0, pi/12, pi/2])
a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
w = 2 * pi
tspan = np.array([0.0, 1.0])
y0 = np.array([-1.0, 0.0, 0.0])

sol = solve_ivp(fun=lambda t, y: odefcn(t, y, a, b, w, events), t_span=tspan, y0=y0)

plt.title('Synthetic ECG')
plt.ylabel('mV')
plt.xlabel('time (s)')
plt.plot(sol.t, sol.y[2], 'b-')

plt.show()