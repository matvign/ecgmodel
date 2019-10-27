#!/usr/bin/env python3
import numpy as np
import sympy as sp


def import_sample():
    csvdata = np.genfromtxt("built.csv", delimiter=",", skip_header=2)
    if csvdata.ndim != 2:
        return None
    if csvdata.shape[0] == 0 or csvdata.shape[1] != 2:
        return None
    data = csvdata[csvdata[:, 0] <= 1]
    return (data[:, 0], data[:, 1])


def defaults():
    a = np.array([12, -50, 300, -75, 7.5])
    b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    evt = np.array([-np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2])
    omega = 2*np.pi
    return (a, b, evt, omega)


def discrete_model(dt, X, omega=2*np.pi, N=0):
    """Alternate version of discrete model where state
    X = [theta, z, a1, .., b1, .., evt1, .., evt5]
    """
    theta, z = X[0:2]
    a = X[2:7]
    b = X[7:12]
    evt = X[12:17]
    dtheta = [(theta - ei) for ei in evt]

    Xk = np.zeros(17)
    Xk[0] = theta + omega*dt
    Xk[1] = z + (N * dt) - sum(
        dt * ai * omega * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    Xk[2:17] = [*a, *b, *evt]
    return Xk


def solve_ecg(a, b, evt, omega=2*np.pi):
    tk, tf = (0, 1)
    ts = []
    dt = 0.008

    xk = np.array([-np.pi, 0, *a, *b, *evt], dtype="float")
    xs = []

    while tk < tf:
        xk = discrete_model(dt, xk, omega)
        xs.append(xk)

        ts.append(tk)
        tk += dt

    return (ts, [i[1] for i in xs])


def state_jacobian(dt, X, omega):
    """Manually defined jacobian of discrete model wrt state vector
    X = [theta, z, a1, .., b1, .., e1, .., e5]
    """
    theta, z = X[0:2]
    a = X[2:7]
    b = X[7:12]
    evt = X[12:17]
    dtheta = [(theta - ei) for ei in evt]

    dF_theta = 1
    dF_z = 0

    dG_theta = -sum((
        dt * ai * omega * (1 - (dthi**2/bi**2)) * np.exp(-(dthi**2)/2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_z = 1
    dG_ak = [
        -dt * omega * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_bk = [
        -2 * dt * (ai/bi**3) * omega * dthi * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_ek = [
        dt * ai * omega * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    A = np.asmatrix(np.eye(17), dtype="float")
    A[0, :2] = [dF_theta, dF_z]
    A[1, :] = [dG_theta, dG_z, *dG_ak, *dG_bk, *dG_ek]
    for i in range(2, 17):
        A[i, i] = 1
    return A


def state_jacobian_():
    """Jacobian of discrete model computed using SymPy wrt state vector
    X = [theta, z, a1, .., b1, .., e1, .., e5]
    """
    dt = sp.symbols("dt")
    theta, z = sp.symbols("theta, z")
    a = sp.symbols("a1, a2, a3, a4, a5")
    b = sp.symbols("b1, b2, b3, b4, b5")
    e = sp.symbols("e1, e2, e3, e4, e5")
    omega = sp.symbols("omega")
    N = sp.symbols("N")
    u = sp.symbols(", ".join(["u{}".format(i) for i in range(1, 16)]))

    w = [omega, N, *u]
    dtheta = [(theta - ei) for ei in e]

    theta_k = theta + omega * dt
    zk = z + (N*dt) - sum((
        dt * ai * omega * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ))
    ak = [(ai + ui) for ai, ui in zip(a, u[:5])]
    bk = [(bi + ui) for bi, ui in zip(b, u[5:10])]
    ek = [(ei + ui) for ei, ui in zip(e, u[10:15])]
    m = sp.Matrix([theta_k, zk, *ak, *bk, *ek])
    state = sp.Matrix([theta, z, *a, *b, *e])
    j = m.jacobian(state)
    func = sp.lambdify([dt, theta, z, *a, *b, *e, *w], j)
    res = lambda dt, xk, omega, w=[0 for i in range(0, 16)]: func(dt, xk, omega, w)
    return res


def noise_jacobian(dt, X, omega=2*np.pi, N=0):
    """Manually defined jacobian of discrete model wrt noise vector
    wk = [omega, n, u1, .., u15]
    """
    theta, z = X[0:2]
    a = X[2:7]
    b = X[7:12]
    evt = X[12:17]
    dtheta = [(theta - ei) for ei in evt]

    dF_omega = dt
    dF_n = 0

    dG_omega = -sum((
        dt * (ai/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_n = dt

    F = np.asmatrix(np.zeros((17, 17)), dtype="float")
    F[0, :2] = [dF_omega, dF_n]
    for i in range(2, 17):
        F[i, i] = 1
    return F


def noise_jacobian_():
    """Jacobian of discrete model computed using SymPy wrt noise vector
    wk = [omega, n, u1, .., u15]
    """
    dt = sp.symbols("dt")
    theta, z = sp.symbols("theta, z")
    a = sp.symbols("a1, a2, a3, a4, a5")
    b = sp.symbols("b1, b2, b3, b4, b5")
    e = sp.symbols("e1, e2, e3, e4, e5")
    omega = sp.symbols("omega")
    N = sp.symbols("N")
    u = sp.symbols(", ".join(["u{}".format(i) for i in range(1, 16)]))

    w = [omega, N, *u]
    dtheta = [(theta - ei) for ei in e]

    theta_k = theta + omega * dt
    zk = z + (N*dt) - sum((
        dt * ai * omega * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ))
    ak = [(ai + ui) for ai, ui in zip(a, u[:5])]
    bk = [(bi + ui) for bi, ui in zip(b, u[5:10])]
    ek = [(ei + ui) for ei, ui in zip(e, u[10:15])]
    m = sp.Matrix([theta_k, zk, *ak, *bk, *ek])
    state = sp.Matrix([*w])
    j = m.jacobian(state)
    func = sp.lambdify([dt, theta, z, *a, *b, *e, *w], j)
    res = lambda xk, dt, omega, w=[0 for i in range(0, 16)]: func(dt, xk, omega, w)
    return res


def parameter_est(ys, ts, a, b, evt, omega):
    """Parameter fit sample ECG with Extended Kalman Filter

    x: state vector
    p: covariance matrix
    w: process noise vector
    v: measurement noise vector

    Q: variance of process noise
    R: variance of measurement noise
    Qk: covariance matrix of process noise w/variance Q
    Rk: covariance matrix of measurement noise w/variance R

    f: system propagation model
    A: jacobian of system wrt state vector x
    F: jacobian of system wrt noise vector w

    g: state-measurement model
    C: jacobian of state-measurement wrt state vector x
    G: jacobian of state-measurement wrt noise vector v

    """
    xk = np.asarray([-np.pi, 0, *a, *b, *evt], dtype="float")
    pk = np.asmatrix(np.eye(17), dtype="float")

    Q = float(1)
    Qk = np.asmatrix(np.eye(17)*Q, dtype="float")
    R = float(1)
    Rk = np.asmatrix(np.eye(1)*R, dtype="float")

    A = state_jacobian_()
    F = noise_jacobian_()

    g = np.asarray([0, 1, *[0 for i in range(15)]])
    C = g
    G = np.asmatrix(np.zeros(17))

    a = np.asarray(a, dtype="float")
    b = np.asarray(b, dtype="float")
    evt = np.asarray(evt, dtype="float")

    t_old = ts[0]
    xs = []

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform state and covariance prediction
        x_hat = discrete_model(xk, dt, omega)
        Ak = state_jacobian(xk, dt, omega)
        Fk = noise_jacobian(xk, dt, omega)
        p_hat = Ak*pk*Ak.T + Fk*Qk*Fk.T

        # perform state update
        C = np.asmatrix([0, 1, *[0 for i in range(15)]], dtype="float")
        G = np.asmatrix([1], dtype="float")
        K = p_hat*C.T * (C*p_hat*C.T + G.T)

        xk = x_hat + K*(yk - g*x_hat)
        xk = np.asarray(xk.T)[0]
        pk = p_hat - K*C*p_hat

        xs.append(xk)

        # update last time
        t_old = tk

    return (ts, [i[2] for i in xs[:]], xs[-1][3:18])


def main():
    np.set_printoptions(suppress=True)
    a, b, evt, omega = defaults()
    ts, ys = import_sample()
    res = parameter_est(ts, ys, a, b, evt, omega)
    return res
