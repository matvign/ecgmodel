#!/usr/bin/env python3
import numpy as np
import sympy as sp

from ecgmodel.helpers.helper import findpeak


def ecg_denoise_model(X, h, a, b, evt, omega=2*np.pi, z0=0, w=[0, 0, 0]):
    """Discrete ecg model used for denoising ecg w/optional process noise w """
    x, y, z = X
    w1, w2, w3 = w
    Xk = np.zeros(3)
    theta = np.arctan2(y, x)
    dtheta = [(theta - ei) for ei in evt]

    alpha = 1.0 - np.sqrt(np.power(x, 2) + np.power(y, 2))

    Xk[0] = (1 + alpha * h)*x - (omega * h * y) + w1
    Xk[1] = (1 + alpha * h)*y + (omega * h * x) + w2
    Xk[2] = -((h-1)*z - h*z0) + w3 - sum(
        ((ai * omega * h * dthi) * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    return Xk


def ecg_denoise_jacobian():
    """Utilizes SymPy to return jacobian function of ecg_denoise_model """
    x, y, z = sp.symbols("x, y, z")
    z0 = sp.symbols("z0")
    h = sp.symbols("h")
    a1, a2, a3, a4, a5 = sp.symbols("a1, a2, a3, a4, a5")
    b1, b2, b3, b4, b5 = sp.symbols("b1, b2, b3, b4, b5")
    e1, e2, e3, e4, e5 = sp.symbols("e1, e2, e3, e4, e5")
    omega = sp.symbols("omega")
    w1, w2, w3 = sp.symbols("w1, w2, w3")

    a = [a1, a2, a3, a4, a5]
    b = [b1, b2, b3, b4, b5]
    evt = [e1, e2, e3, e4, e5]
    dtheta = [(sp.atan2(y, x) - ei) for ei in evt]

    alpha = 1 - sp.sqrt(x**2 + y**2)
    F = (1 + alpha*h)*x - (omega * h * y) + (w1 * h)
    G = (1 + alpha*h)*y + (omega * h * x) + (w2 * h)
    H = -((h-1)*z - h*z0) + (w3 * h) - sum(
        ((ai * omega * h * dthi) * sp.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )

    state = sp.Matrix([x, y, z])
    j = sp.Matrix([F, G, H]).jacobian(state)
    return sp.lambdify([x, y, z, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5,
                e1, e2, e3, e4, e5, omega, w1, w2, w3, h], j)


def ecg_denoise_predict(jacobian, dt, X, P, Q, a, b, evt, omega):
    """Function for ECG predict step of covariance matrix
    This function uses a fixed a, b, evt and w
    """
    # mu, sigma = 0, 0.1
    # w = np.random.normal(mu, sigma, 3)
    A = jacobian(*X, *a, *b, *evt, omega, *[0, 0, 0], dt)
    F = np.asmatrix(np.eye(3), dtype="float")
    priori_p = A*P*A.T + F*Q*F.T
    return priori_p


def ecg_denoise_update(yk, dt, X, P):
    X = np.asmatrix(X, dtype="float").T
    r = np.random.normal(0, 0.005, 1)

    g = np.matrix([0, 0, 1], dtype="float") * X + r

    C = np.matrix([0, 0, 1-dt], dtype="float")
    G = np.matrix([1], dtype="float")

    s = C*P*C.T + G
    K = P*C.T * s.I

    posteri_x = X + K * (yk-g)
    posteri_p = P - K*C*P
    return (np.array(posteri_x.T)[0], posteri_p)


def denoise_ecg_ekf(ys, ts, a, b, evt, omega):
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
    try:
        omega = findpeak(ts, ys)
    except:
        omega = 2 * np.pi

    xk = np.asarray([-1, 0, 0], dtype="float")
    pk = np.asmatrix(np.eye(3), dtype="float")
    Q = np.asmatrix(np.eye(3), dtype="float")
    jacobian_f = ecg_denoise_jacobian()

    t_old = ts[0]
    xs = []

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform state prediction
        x_hat = ecg_denoise_model(xk, dt, a, b, evt, omega)

        # perform covariance prediction
        p_hat = ecg_denoise_predict(jacobian_f, dt, xk, pk, Q, a, b, evt, omega)

        # perform state update
        xk, pk = ecg_denoise_update(yk, dt, x_hat, p_hat)
        xs.append(xk)

        # update last time
        t_old = tk

    return (ts, [i[2] for i in xs[:]])


def denoise_ecg_ekf_gen(ys, ts, a, b, evt, omega):
    """Representation of denoise_ecg_ekf as a generator function for potential
    step-by-step Kalman Filter
    """
    try:
        omega = findpeak(ts, ys)
    except:
        omega = 2 * np.pi

    xk = np.asarray([-1, 0, 0], dtype="float")
    pk = np.asmatrix(np.eye(3), dtype="float")
    Q = np.asmatrix(np.eye(3), dtype="float")
    jacobian_f = ecg_denoise_jacobian()

    t_old = ts[0]
    xs = []

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform state prediction
        x_hat = ecg_denoise_model(xk, dt, a, b, evt, omega)

        # perform covariance prediction
        p_hat = ecg_denoise_predict(jacobian_f, dt, xk, pk, Q, a, b, evt, omega)

        # perform state update
        xk, pk = ecg_denoise_update(yk, dt, x_hat, p_hat)
        xs.append(xk)

        # update last time
        t_old = tk
        yield xk

    return (ts, [i[2] for i in xs[:]])
