#!/usr/bin/env python3
import numpy as np
import sympy as sp

from ecgmodel.helpers import helper


def defaults():
    a = np.array([12, -50, 300, -75, 7.5])
    b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    evt = np.array([-np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2])
    omega = 2*np.pi
    return (a, b, evt, omega)


def state_jacobian(dt, X, omega, z0=0):
    """F matrix for EKF """
    x, y, z = X[0:3]
    a = X[3:8]
    b = X[8:13]
    e = X[13:18]
    # omega = X[18]

    theta = np.arctan2(y, x)
    dtheta = [(theta - ei) for ei in e]
    sq_xy = x**2 + y**2
    sqrt_xy = np.sqrt(sq_xy)

    dX_x = 1+dt - ((2*dt*x**2 + dt*y**2)/sqrt_xy)
    dX_y = -omega*dt - ((dt*x*y)/sqrt_xy)
    dX_z = 0

    dY_x = omega*dt - ((dt*x*y)/sqrt_xy)
    dY_y = 1+dt - ((dt*x**2 + 2*dt*y**2)/sqrt_xy)
    dY_z = 0

    dZ_x = sum((
        (((ai/bi**2)*omega*dt*y)/sq_xy) * np.exp(-(dthi**2)/(2*bi**2)) * (1 - (dthi**2/bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dZ_y = sum((
        ((-(ai/bi**2)*omega*dt*x)/sq_xy) * np.exp(-(dthi**2)/(2*bi**2)) * (1 - (dthi**2/bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    dZ_z = 1-dt
    dZ_ak = [
        -dt * omega * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dZ_bk = [
        -(a/(b**3))*dt*omega*dthi**3 * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dZ_ek = [
        dt * (ai/bi**2) * omega * (1 - ((dthi**2)/(bi**2))) * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    ]
    dZ_omega = - sum((
        dt * (ai/bi**2) * dthi * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )

    F = np.asmatrix(np.zeros((18, 18)), dtype="float")
    F[0, :3] = [dX_x, dX_y, dX_z]
    F[1, :3] = [dY_x, dY_y, dY_z]
    F[2, :18] = [dZ_x, dZ_y, dZ_z, *dZ_ak, *dZ_bk, *dZ_ek]
    for i in range(3, 18):
        F[i, i] = 1
    return F


def state_jacobian2():
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

    F = (1 + alpha*dt)*x - omega*dt*y
    G = (1 + alpha*dt)*y + omega*dt*x
    H = -((dt-1)*z - (dt*z0)) - sum(
        ai * omega * dt * dthi * sp.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    m = sp.Matrix([F, G, H, *a, *b, *e])
    state = sp.Matrix([x, y, z, *a, *b, *e])
    j = m.jacobian(state)
    return sp.lambdify([dt, x, y, z, *a, *b, *e, omega, z0], j)


def parameter_est(ts, ys, a, b, evt, omega, z0=0):
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
    xk = np.array([-1, 0, 0, *a, *b, *evt])
    pk = np.asmatrix(np.eye(18)*1, dtype="float")

    Q = float(1)
    R = float(1)
    Qk = np.asmatrix(np.eye(18)*Q, dtype="float")
    Rk = np.asmatrix([R])

    xs = []
    t_old = ts[0]

    F = state_jacobian2()

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform prediction
        x_hat = helper.discrete_ecg_model(dt, xk, omega)
        x_hat = np.asmatrix(x_hat).T

        # Fk = state_jacobian(dt, xk, omega, z0)
        Fk = F(dt, *xk, omega, z0)
        p_hat = Fk * pk * Fk.T + Qk
        print("-- Prediction --")
        print("-- tk:", tk, "dt:", dt)
        print("-- x_hat --\n", x_hat.T)
        # print("-- p_hat --\n", p_hat)
        # print("-- F --\n", Fk)
        # print("-- Q --\n", Qk)
        print()

        # perform update
        A = np.matrix([0, 0, 1, *[0 for i in range(0, 15)]])
        pk = (p_hat.I + A.T*Rk.I*A).I
        K = pk * A.T * Rk.I
        xk = x_hat + K * (yk - A*x_hat)
        xk = np.asarray(xk.T, dtype="float")[0]
        xs.append(xk)
        print("-- Update --")
        print("-- xk --\n", xk.T)
        # print("-- pk --\n", pk)
        # print("-- A --\n", A)
        # print("-- Rk.I --\n", Rk.I)
        print()

        # yield xk

        # set old time to current time
        t_old = tk

    print("end")
    return (ts, [i[2] for i in xs[:]], xs[-1][3:18])


def main():
    np.set_printoptions(suppress=True)
    a, b, evt, omega = defaults()
    ts, ys = helper.import_sample()
    res = parameter_est(ts, ys, a, b, evt, omega)
    print(res[2])
    return res
