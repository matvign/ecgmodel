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

    F: state transition model
    A: measurement function

    Q: variance for process noise
    R: variance for measurement noise
    Qk: matrix for process noise
    Rk: matrix for measurement noise

    G: control matrix for external influence, nothing in this case
    u: control vector for external influence, nothing in this case

    """
    xk = np.array([-1, 0, 0, *a, *b, *evt, omega])

    P = opts
    pk = np.asmatrix(np.eye(19)*P, dtype="float")

    Q = float(1)
    R = float(1)
    Qk = np.asmatrix(np.eye(19)*Q, dtype="float")
    Rk = np.asmatrix([R])

    xs = []
    t_old = ts[0]

    F = state_jacobian()

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform prediction
        x_hat = helper.discrete_ecg_model(dt, xk)
        x_hat = np.asmatrix(x_hat).T

        # Fk = state_jacobian(dt, xk, omega, z0)
        Fk = F(dt, *xk, z0)
        p_hat = Fk * pk * Fk.T + Qk
        # print("-- Prediction --")
        # print("-- tk:", tk, "dt:", dt)
        # print("-- x_hat --\n", x_hat.T)
        # print("-- p_hat --\n", p_hat)
        # print("-- F --\n", Fk)
        # print("-- Q --\n", Qk)
        # print()

        # perform update
        h = np.matrix([0, 0, 1, *[0 for i in range(16)]])
        H = np.matrix([0, 0, 1-dt, *[0 for i in range(16)]])

        zk = yk - h*x_hat
        S = H * p_hat * H.T + Rk
        K = p_hat * H.T * S.I
        pk = p_hat - K*H*p_hat

        xk = x_hat + K*zk
        xk = np.array(xk.T, dtype="float")[0]
        xs.append(xk)
        # print("-- Update --")
        # print("-- xk --\n", xk.T)
        # print("-- pk --\n", pk)
        # print("-- A --\n", A)
        # print("-- Rk.I --\n", Rk.I)
        # print()

        # yield xk

        # set old time to current time
        t_old = tk

    print("end")
    return (ts, [i[2] for i in xs[:]], xs[-1][3:19])
