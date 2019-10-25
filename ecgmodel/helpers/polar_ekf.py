#!/usr/bin/env python3
import numpy as np


def polar_discrete_model(X, a, b, evt, dt, omega=2*np.pi, N=0):
    theta, z = X
    dX = np.zeros(2)
    dtheta = [(theta - ei) for ei in evt]

    dX[0] = theta + omega * dt
    dX[1] = z + (N * dt) - sum(
        dt * omega * (ai/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    return dX


def polar_state_jacobian(X, a, b, evt, dt, omega=2*np.pi):
    """Polar ECG model jacobian wrt state variables theta, z """
    theta, z = X
    dtheta = [(theta - ei) for ei in evt]

    dF_theta = 1
    dF_z = 0

    dG_theta = -sum((
        dt * omega * (ai/bi**2) * (1 - (dthi**2/bi**2)) * np.exp(-(dthi**2)/2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_z = 1
    return np.matrix([
        [dF_theta, dF_z],
        [dG_theta, dG_z]
    ])


def polar_state_jacobian2(X, a, b, evt, omega, dt):
    """Polar ECG model jacobian wrt state variables
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
        -2 * dt * (ai/bi**3) * omega * (1 - (dthi**2/2*bi**2)) * np.exp(-(dthi**2)/(2*bi**2))
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


def polar_noise_jacobian(X, a, b, evt, dt, omega=2*np.pi, N=0):
    """Polar ECG model jacobian wrt process noise
    wk = [a1, .., a5, b1, .., b5, theta1, .., theta5, omega, N]
    """
    theta, z = X
    dtheta = [(theta - ei) for ei in evt]

    dF_ak = [0 for i in range(0, 5)]
    dF_bk = [0 for i in range(0, 5)]
    dF_ek = [0 for i in range(0, 5)]
    dF_omega = dt
    dF_n = 0

    dG_ak = [
        -dt * omega * (dthi/bi**2) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_bk = [
        2 * dt * omega * dthi * (ai/bi**3) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_ek = [
        dt * omega * (ai/bi**2) * (1 - (dthi**2)/b**2) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dG_omega = -sum((
        dt * dthi * (ai/bi**2) * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dG_n = 1
    return np.matrix([
        [*dF_ak, *dF_bk, *dF_ek, dF_omega, dF_n],
        [*dG_ak, *dG_bk, *dG_ek, dG_omega, dG_n]
    ])


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
        xk = polar_discrete_model(xk, a, b, evt, dt, omega)
        xs.append(xk)

        ts.append(tk)
        tk += dt

    return (ts, [i[1] for i in xs])


def solve_ecg_ekf(ys, ts, a, b, evt, omega):
    """Run Kalman Filter

    ys: observations
    ts: the timeframes
    xk: vector for the system state
    pk: covariance matrix of the system
    w: process noise vector
    v: measurement noise vector

    f(): ecg model
    g(): update function
    A: jacobian of system of equations wrt x
    F: jacobian of system of equations wrt w
    C: jacobian of update equations wrt x
    G: jacobian of update equations wrt v
    Q: process noise matrix, equal to F * Q * F.T
    R: measurement noise matrix, equal to G * R * G.T

    r: innovation signal
    k: kalman gain

    a: values that only change during update step
    b: values that only change during update step
    evt: values that only change during update step
    omega: angular velocity, fixed in our case

    """
    xk = np.asarray([-np.pi/3, 0], dtype="float")
    pk = np.asmatrix(np.eye(2), dtype="float")

    Q = np.asmatrix(np.eye(17), dtype="float")
    jacobian_f = polar_discrete_model()

    a = np.asarray(a, dtype="float")
    b = np.asarray(b, dtype="float")
    evt = np.asarray(evt, dtype="float")

    t_old = ts[0]
    xs = []

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform state prediction
        # x_hat = ecg_discrete_model(xk, dt, a, b, evt, omega)
        x_hat = ecg_discrete_polar(xk, dt, a, b, evt, omega)

        # perform covariance prediction
        # p_hat = ecg_predict(jacobian_f, dt, xk, pk, Q, a, b, evt, omega)
        p_hat = ecg_predict2(jacobian_f, xk, dt, pk, Q, dt, a, b, evt)

        # perform state update
        xk, pk = ecg_update2(yk, dt, x_hat, p_hat)
        xs.append(xk)

        # update last time
        t_old = tk

    # return (ts, [i[2] for i in xs[:]])
    return (ts)


def ecg_predict(jacobian, X, dt, P, Q, a, b, evt):
    """Function for ECG predict step of covariance matrix
    This function uses a fixed a, b, evt and w
    """
    A = jacobian(*X, *a, *b, *evt, dt)
    F = np.asmatrix(np.eye(17), dtype="float")
    priori_p = A*P*A.T + F*Q*F.T
    return priori_p


def ecg_update(yk, X, P, R):
    X = np.asmatrix(X, dtype="float").T
    r = np.asmatrix(np.random.normal(0, 0.1, 2)).T

    g = np.matrix(np.eye(2), dtype="float") * X + r

    C = np.matrix(np.eye(2), dtype="float")
    G = np.matrix([1, 1], dtype="float").T

    s = C*P*C.T + G
    K = P*C.T * s.I

    posteri_x = X + K * (yk-g)
    posteri_p = P - K*C*P
    return (np.array(posteri_x.T)[0], posteri_p)
