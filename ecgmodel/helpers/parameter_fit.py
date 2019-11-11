#!/usr/bin/env python3
import numpy as np
import sympy as sp

from ecgmodel.helpers import helper


def state_jacobian():
    """Return function for jacobian of state
    X = [x, y, z, a1, .., a5, b1, .., b5, e1, .., e5]
    """
    x, y, z, z0 = sp.symbols("x, y, z, z0")
    omega = sp.symbols("omega")
    dt = sp.symbols("dt")

    a = list(sp.symbols("a1, a2, a3, a4, a5"))
    b = list(sp.symbols("b1, b2, b3, b4, b5"))
    e = list(sp.symbols("e1, e2, e3, e4, e5"))

    alpha = 1 - sp.sqrt(x**2 + y**2)
    theta = sp.atan2(y, x)
    dtheta = [(theta - ei) for ei in e]

    F = (1 + alpha*dt)*x - (omega*dt*y)
    G = (1 + alpha*dt)*y + (omega*dt*x)
    H = -((dt-1)*z - (dt*z0)) - sum(
        ai * omega * dt * dthi * sp.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    m = sp.Matrix([F, G, H, *a, *b, *e, omega])
    state = sp.Matrix([x, y, z, *a, *b, *e, omega])
    j = m.jacobian(state)
    return sp.lambdify([dt, x, y, z, *a, *b, *e, omega, z0], j)


def parameter_est(ts, ys, a, b, evt, omega, opts, z0=0):
    """Perform parameter estimation of ECG sample with EKF

    x: state vector
    p: covariance matrix
    Q: variance for process noise
    Qk: covariance matrix for process noise
    R: variance for measurement noise
    Rk: covariance matrix for measurement noise

    f: process function
    w: process noise vector
    A: jacobian of f wrt x
    F: jacobian of f wrt w, this can be set to identity matrix

    g: measurement function
    v: measurement noise vector
    C: jacobian of g wrt x
    G: jacobian of g wrt v
    """
    xk = np.array([-1, 0, 0, *a, *b, *evt, omega])
    P = opts
    pk = np.asmatrix(np.eye(19)*P, dtype="float")

    Q = float(1)
    Qk = np.asmatrix(np.eye(19)*Q, dtype="float")

    R = float(1)
    Rk = np.matrix([1])

    A = state_jacobian()

    xs = []
    t_old = ts[0]
    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform prediction
        x_hat = helper.discrete_ecg_model(dt, xk)
        x_hat = np.asmatrix(x_hat).T

        Ak = A(dt, *xk, z0)
        p_hat = Ak * pk * Ak.T + Qk
        # print("-- Prediction --")
        # print("-- tk:", tk, "dt:", dt)
        # print("-- x_hat --\n", x_hat.T)
        # print("-- p_hat --\n", p_hat)
        # print("-- A --\n", Fk)
        # print("-- F --\n", Qk)
        # print()

        # perform update
        g = np.matrix([0, 0, 1, *[0 for i in range(16)]])
        C = np.matrix([0, 0, 1-dt, *[0 for i in range(16)]])

        vk = np.random.normal(0, 0, 1)
        gk = g*x_hat + vk

        zk = yk - gk
        S = C * p_hat * C.T + Rk
        K = p_hat * C.T * S.I
        pk = p_hat - K*C*p_hat

        xk = x_hat + K*zk
        xk = np.array(xk.T, dtype="float")[0]
        xs.append(xk)
        # print("-- Update --")
        # print("-- xk --\n", xk.T)
        # print("-- pk --\n", pk)
        # print("-- C --\n", C)
        # print("-- G --\n", G)
        # print()

        # yield xk

        # set old time to current time
        t_old = tk

    print("end")
    return (ts, [i[2] for i in xs[:]], xs[-1][3:19])
