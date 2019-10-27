#!/usr/bin/env python3
import numpy as np


def discrete_model(X, a, b, evt, dt, omega=2*np.pi, N=0):
    theta, z = X
    dX = np.zeros(2)
    dtheta = [(theta - ei) for ei in evt]

    dX[0] = theta + omega * dt
    dX[1] = z + (N * dt) - sum(
        dt * omega * (ai/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    return dX


def discrete_model_alt(X, dt, omega=2*np.pi, N=0):
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
        dt * omega * ai * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    Xk[2:17] = [*a, *b, *evt]
    return Xk


def state_jacobian(X, a, b, evt, dt, omega):
    """Function to return jacobian of polar ECG model wrt state vector
    X = [theta, z, a1, .., a5, b1, .., b5, theta1, .., theta5, omega, N]

    """
    theta, z = X
    dtheta = [(theta - ei) for ei in evt]

    dF_theta = 1
    dF_z = 0

    dG_theta = -sum((
        dt * (ai/bi**2) * omega * (1 - (dthi**2/bi**2)) * np.exp(-(dthi**2)/2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_z = 1
    dG_ak = [
        -dt * (omega/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_bk = [
        -2 * dt * (ai/bi**3) * omega * dthi * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_ek = [
        dt * (ai/bi**2) * omega * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    A = np.asmatrix(np.eye(17), dtype="float")
    A[0, :2] = [dF_theta, dF_z]
    A[1, :] = [dG_theta, dG_z, *dG_ak, *dG_bk, *dG_ek]
    for i in range(2, 17):
        A[i, i] = 1
    return A


def state_jacobian_alt(X, dt, omega):
    """Alternative to state_jacobian() """
    theta, z = X[0:2]
    a = X[2:7]
    b = X[7:12]
    evt = X[12:17]
    dtheta = [(theta - ei) for ei in evt]

    dF_theta = 1
    dF_z = 0

    dG_theta = -sum((
        dt * (ai/bi**2) * omega * (1 - (dthi**2/bi**2)) * np.exp(-(dthi**2)/2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_z = 1
    dG_ak = [
        -dt * (omega/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_bk = [
        -2 * dt * (ai/bi**3) * omega * dthi * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_ek = [
        dt * (ai/bi**2) * omega * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    A = np.asmatrix(np.eye(17), dtype="float")
    A[0, :2] = [dF_theta, dF_z]
    A[1, :] = [dG_theta, dG_z, *dG_ak, *dG_bk, *dG_ek]
    for i in range(2, 17):
        A[i, i] = 1
    return A


def noise_jacobian(X, a, b, evt, dt, omega=2*np.pi, N=0):
    """Function to return jacobian of polar ECG model wrt noise vector
    wk = [omega, n, u1, .., u15]

    """
    theta, z = X
    dtheta = [(theta - ei) for ei in evt]

    dF_omega = dt
    dF_n = 0

    dG_omega = -sum((
        dt * (ai/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_n = 1

    F = np.asmatrix(np.zeros(17, 17))
    F[0, :2] = [dF_omega, dF_n]
    for i in range(2, 17):
        F[i, i] = 1
    return F


def noise_jacobian_alt(X, dt, omega=2*np.pi, N=0):
    """Function to return jacobian of polar ECG model wrt noise vector
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
    dG_n = 1

    F = np.asmatrix(np.zeros(17, 17))
    F[0, :2] = [dF_omega, dF_n]
    for i in range(2, 17):
        F[i, i] = 1
    return F


def solve_ecg(a, b, evt, omega=2*np.pi):
    arr_a = np.asarray(a, dtype="float")
    arr_b = np.asarray(evt, dtype="float")
    arr_evt = np.asarray(evt, dtype="float")

    tk, tf = (0, 1)
    ts = []
    dt = 0.008

    xk = np.array([-np.pi, 0], dtype="float")
    xs = []

    while tk < tf:
        xk = discrete_model(xk, a, b, evt, dt, omega)
        xs.append(xk)

        ts.append(tk)
        tk += dt

    return (ts, [i[1] for i in xs])


def solve_ecg_ekf(ys, ts, a, b, evt, omega):
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

    # f = discrete_model_alt
    # A = state_jacobian_alt
    # F = noise_jacobian

    g = np.asarray([0, 1, *[0 for i in range(15)]])
    # C = np.asmatrix([0, 1])
    # G = np.asmatrix([1])

    a = np.asarray(a, dtype="float")
    b = np.asarray(b, dtype="float")
    evt = np.asarray(evt, dtype="float")

    t_old = ts[0]
    xs = []

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform state and covariance prediction
        x_hat = discrete_model_alt(xk, dt, omega)
        Ak = state_jacobian_alt(xk, dt, omega)
        Fk = noise_jacobian_alt(xk, dt, omega)
        p_hat = Ak * pk * Ak.T + Fk * Qk * Fk.T

        # perform state update
        C = np.asmatrix([0, 1], dtype="float")
        G = np.asmatrix([1], dtype="float")
        K = p_hat * C.T * (C * p_hat * C.T + G.T)

        xk = x_hat + K * (yk - g*x_hat)
        xk = np.asarray(xk.T)[0]
        pk = p_hat - K*C*p_hat

        xs.append(xk)

        # update last time
        t_old = tk

    # return (ts, [i[2] for i in xs[:]])
    return (ts)
