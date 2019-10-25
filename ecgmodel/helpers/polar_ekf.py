#!/usr/bin/env python3
import numpy as np
import sympy as sp


def ecg_discrete_polar(X, a, b, evt, dt, omega=2*np.pi, N=0):
    theta, z = X
    dX = np.zeros(2)
    dtheta = [(theta - ei) for ei in evt]

    dX[0] = theta + omega * dt
    dX[1] = z + (N * dt) - sum(
        (ai * dt * omega * dthi) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    return dX


def ecg_polar_state_jacobian():
    """Jacobian function for process model wrt state """
    theta, z = sp.symbols("theta, z")
    a1, a2, a3, a4, a5 = sp.symbols("a1, a2, a3, a4, a5")
    b1, b2, b3, b4, b5 = sp.symbols("b1, b2, b3, b4, b5")
    e1, e2, e3, e4, e5 = sp.symbols("e1, e2, e3, e4, e5")

    a = [a1, a2, a3, a4, a5]
    b = [b1, b2, b3, b4, b5]
    e = [e1, e2, e3, e4, e5]
    dth = [(theta - ei) for ei in e]
    h = sp.symbols("h")

    dG_theta = sum(
        (-h * ai * (1 - (dthi**2/b**2)) * sp.exp(-dthi**2/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dth)
    )

    m = sp.matrix([
        [1, 0],
        [dG_theta, 1]
    ])
    return sp.lambdify([theta, z, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5,
                        e1, e2, e3, e4, e5, h], m)


def ecg_polar_noise_jacobian():
    """Jacobian function for process model wrt noise """
    theta, z = sp.symbols("theta, z")
    w = sp.symbols("w")
    a1, a2, a3, a4, a5 = sp.symbols("a1, a2, a3, a4, a5")
    b1, b2, b3, b4, b5 = sp.symbols("b1, b2, b3, b4, b5")
    e1, e2, e3, e4, e5 = sp.symbols("e1, e2, e3, e4, e5")
    N = sp.symbols("N")

    a = [a1, a2, a3, a4, a5]
    b = [b1, b2, b3, b4, b5]
    e = [e1, e2, e3, e4, e5]
    dth = [(theta - ei) for ei in e]
    h = sp.symbols("h")

    dG_a = [(-h * dthi * sp.exp(-dthi**2/2*bi**2)) for bi, dthi in zip(b, dth)]
    dG_b = [(-h * ai * (dthi**3/bi**3) * sp.exp(-dthi**2/(2*bi**2))
             for ai, bi, dthi in zip(a, b, dth))]
    dG_e = [(h * ai * (1 - (dthi**2/bi**2)) * sp.exp(-dthi**2/(2*bi**2)))
            for ai, bi, dthi in zip(a, b, dth)]
    m = sp.matrix([0, *dG_a, *dG_b, *dG_e, 0, h])
    return sp.lambdify([theta, z, a1, a2, a3, a4, a5, e1, e2, e3, e4, e5,
                        b1, b2, b3, b4, b5, w, N, h])


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
    jacobian_f = ecg_polar_state_jacobian()

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


def solve_ecg(a, b, evt, omega=2*np.pi):
    arr_a = np.asarray(a, dtype="float")
    arr_b = np.asarray(evt, dtype="float")
    arr_evt = np.asarray(evt, dtype="float")

    tk, tf = (0, 1)
    ts = []
    dt = 0.008

    xk = np.array([-np.pi, 0], dtype="float")
    xs = []

    f = ecg_discrete_polar

    while tk < tf:
        xk = f(xk, a, b, evt, dt, omega)
        xs.append(xk)

        ts.append(tk)
        tk += dt

    return (ts, [i[1] for i in xs])


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
