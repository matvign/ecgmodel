#!/usr/bin/env python3
import json
import re

import numpy as np
import sympy as sp
import numexpr as ne
from scipy.integrate import solve_ivp


pi = np.pi
pi_regx = re.compile(r"(pi\s*|[0-9.]+)(?=(\s*pi|[0-9.]))")


def convert_pi(val):
    def _pirepl(word):
        def repl(matchobj):
            if matchobj.group(1).strip() == "pi":
                return "{}*".format(matchobj.group(1))
            elif matchobj.group(2).strip() == "pi":
                return "{}*".format(matchobj.group(1))
            else:
                return matchobj.group(0)
        return re.sub(pi_regx, repl, word)

    val = ne.evaluate(_pirepl(val), {"pi": np.pi})
    return float(np.asscalar(val))


def import_json(file):
    with open(file) as f:
        data = json.load(f)
        validate_key = ["a", "b", "evt", "omega"]
        validate_len = ["a", "b", "evt"]
        if not all(k in data for k in validate_key):
            return None
        if sum(1 for k in validate_len if len(data[k]) != 5):
            return None
        return data


def export_json(filename, a, b, evt, omega):
    data = {"a": a, "b": b, "evt": evt, "omega": [omega]}
    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)


def import_csv(file):
    csvdata = np.genfromtxt(file, delimiter=",", skip_header=2)
    if csvdata.ndim != 2:
        return None
    if csvdata.shape[0] == 0 or csvdata.shape[1] != 2:
        return None
    return csvdata


def filter_timeframe(data, timeframe):
    return data[data[:, 0] < timeframe]


def ecg_model(X, a, b, evt, omega=2*np.pi, z0=0):
    """Function to solve ODE with scipy.integrate.solve_ivp()
    Modified with omega to increase altitude of z
    Details located here (p291):
    http://web.mit.edu/~gari/www/papers/ieeetbe50p289.pdfModified
    """
    x, y, z = X
    dX = np.zeros(3)
    theta = np.arctan2(y, x)
    dtheta = [(theta - ei) for ei in evt]

    alpha = 1.0 - np.sqrt(np.power(x, 2) + np.power(y, 2))
    dX[0] = (alpha*x) - (omega*y)
    dX[1] = (alpha*y) + (omega*x)
    dX[2] = -(z - z0) - sum(
        ((ai * omega * dthi) * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    return dX


def ecg_discrete_model(X, h, a, b, evt, omega=2*np.pi, z0=0, w=[0, 0, 0]):
    """Discrete form for euler method. Optional process noise in w """
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


def ecg_discrete_jacobian():
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
    d = [(sp.atan2(y, x) - ei) for ei in evt]

    alpha = 1 - sp.sqrt(x**2 + y**2)
    F = (1 + alpha*h)*x - (omega * h * y) + (w1 * h)
    G = (1 + alpha*h)*y + (omega * h * x) + (w2 * h)
    H = -((h-1)*z - h*z0) + (w3 * h) - sum(
        ((ai * omega * h * dthi) * sp.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, d)
    )

    state = sp.Matrix([x, y, z, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5,
                       e1, e2, e3, e4, e5])

    m = sp.Matrix([F, G, H, *[0 for i in range(0, 15)]])
    j = m.jacobian(state)
    return sp.lambdify([x, y, z, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5,
                e1, e2, e3, e4, e5, omega, w1, w2, w3, h], j)


def solve_ecg(a, b, evt, omega):
    arr_a = np.asarray(a, dtype="float")
    arr_b = np.asarray(b, dtype="float")
    arr_evt = np.asarray(evt, dtype="float")

    tspan = (0, 1)
    y0 = np.array([-1, 0, 0])
    teval = np.linspace(0, 1, num=100)
    print("building...")
    fun = lambda t, y: ecg_model(y, a, b, evt, omega)
    sol = solve_ivp(fun=fun, t_span=tspan, y0=y0, t_eval=teval)
    return sol


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
    a = np.asarray(a, dtype="float")
    b = np.asarray(b, dtype="float")
    evt = np.asarray(evt, dtype="float")

    xk = np.asarray([-1, 0, 0, *a, *b, *evt], dtype="float")
    pk = np.asmatrix(np.eye(18), dtype="float")
    Q = np.asmatrix(np.eye(18), dtype="float")
    jacobian_f = ecg_discrete_jacobian()

    t_old = ts[0]
    xs = []

    for tk, yk in zip(ts, ys):
        dt = tk - t_old

        # perform state prediction
        x_hat = ecg_discrete_model(xk[0:3], dt, a, b, evt, omega)
        x_hat = [*x_hat, *a, *b, *evt]
        print(x_hat)

        # perform covariance prediction
        p_hat = ecg_predict(jacobian_f, dt, xk, pk, Q, a, b, evt, omega)

        # perform state update
        xk, pk = ecg_update(yk, dt, x_hat, p_hat)
        xs.append(xk)

        # update last time
        t_old = tk

    return (ts, [i[2] for i in xs[:]])


def ecg_predict(jacobian, tk, X, P, Q, a, b, evt, omega):
    """Function for ECG predict step of covariance matrix
    This function uses a fixed a, b, evt and w
    """
    # mu, sigma = 0, 0.1
    # w = np.random.normal(mu, sigma, 3)
    A = jacobian(*X, omega, *[0, 0, 0], tk)
    F = np.asmatrix(np.eye(18), dtype="float")
    priori_p = A*P*A.T + F*Q*F.T
    return priori_p


def ecg_update(yk, h, X, P):
    X = np.asmatrix(X, dtype="float").T
    r = np.random.normal(0, 0.005, 1)

    g = np.matrix([0, 0, 1, *[0 for i in range(0, 15)]], dtype="float") * X + r

    C = np.matrix([0, 0, 1-h, *[0 for i in range(0, 15)]], dtype="float")
    G = np.matrix([1], dtype="float")

    s = C*P*C.T + G
    K = P*C.T * s.I

    posteri_x = X + K * (yk-g)
    posteri_p = P - K*C*P
    return (np.array(posteri_x.T)[0], posteri_p)
