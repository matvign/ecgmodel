#!/usr/bin/env python3
import json
import re

import numpy as np
from scipy.integrate import solve_ivp

pi = np.pi
pi_regx = re.compile(r"(pi\s*|[0-9.]+)(?=(\s*pi|[0-9.]))")


def pirepl(word):
    def repl(matchobj):
        if matchobj.group(1).strip() == 'pi':
            return "{}*".format(matchobj.group(1))
        elif matchobj.group(2).strip() == 'pi':
            return "{}*".format(matchobj.group(1))
        else:
            return matchobj.group(0)

    return re.sub(pi_regx, repl, word)


def import_json(file):
    with open(file) as f:
        data = json.load(f)
        validate_key = ['a', 'b', 'evt', 'omega']
        validate_len = ['a', 'b', 'evt']
        if not all(k in data for k in validate_key):
            return None
        if sum(1 for k in validate_len if len(data[k]) != 5):
            return None
        return data


def export_json(filename, a, b, evt, omega):
    data = {'a': a, 'b': b, 'evt': evt, 'omega': [omega]}
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def import_csv(file):
    csvdata = np.genfromtxt(file, delimiter=',', skip_header=2)
    if csvdata.ndim != 2:
        return None
    if csvdata.shape[0] == 0 or csvdata.shape[1] != 2:
        return None
    return (csvdata, np.amax(csvdata[:, 0]))
    # return csvdata[csvdata[:, 0] < timeframe]


def filter_timeframe(data, timeframe):
    return data[data[:, 0] < timeframe]


def solve_ecg(a, b, evt, w=2*np.pi):
    arr_a = np.asarray(a, dtype="float")
    arr_b = np.asarray(b, dtype="float")
    arr_evt = np.asarray(evt, dtype="float")

    tspan = np.array([0, 1], dtype="float")
    y0 = np.array([-1.0, 0.0, 0.0], dtype="float")
    teval = np.linspace(0, 1, num=100)
    print('building...')
    sol = solve_ivp(fun=lambda t, y: ecg_model(y, a, b, evt, w),
                    t_span=tspan, y0=y0, t_eval=teval)
    return sol


def ecg_model(X, a, b, evt, w=2*np.pi, z0=0):
    '''
    Function to solve ODE with scipy.integrate.solve_ivp()
    Details located here (p291):
    http://web.mit.edu/~gari/www/papers/ieeetbe50p289.pdf
    '''
    x, y, z = X
    dX = np.zeros(3)

    theta = np.arctan2(y, x)
    alpha = 1.0 - np.sqrt(x**2 + y**2)
    dX[0] = alpha*x - w*y
    dX[1] = alpha*y + w*x

    dX[2] = -(z - z0)
    for i in range(0, 5):
        dtheta = theta - evt[i]
        dX[2] -= a[i] * dtheta * np.exp(-(dtheta**2) / (2 * b[i]**2))

    return dX


def solve_ekf(Y, X, P, Q, R, a, b, evt, w):
    '''
    Y: measurement/observations
    X: state matrix         P: covariance matrix
    Q: measurement noise    R: process noise

    a: constant             b: constant
    evt: constant           w: angular velocity

    Run Kalman Filter until we're out of observations.
    Predict: run predict with the updated state
    Update: create a new update with the estimate
    '''
    Y = np.asarray(Y, dtype="float")
    X, P = np.asmatrix(X, dtype="float").T, np.asmatrix(np.eye(3)*P, dtype="float")
    Q, R = np.asmatrix(Q, dtype="float"), np.asmatrix(np.eye(3)*R, dtype="float")

    a, b = np.asarray(a, dtype="float"), np.asarray(b, dtype="float")
    evt = np.asarray(evt, dtype="float")

    prior_X, prior_P = None, None
    post_X, post_P = X, P
    res = [(post_X, post_P)]

    for obs in Y:
        prior_X, prior_P = predict(post_X, post_P, Q, a, b, evt, w)
        post_X, post_P = update(prior_X, prior_P, R, obs)
        res.append(post_X)

    return res


def ecg_jacobian(X, a, b, evt, w):
    '''
    Linearized version of ecg_model wrt x, y, z
    '''
    x, y, z = np.array(X)[0]
    sqrt_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    dF_x = 1 + (-2*x**2 - y**2)/sqrt_xy
    dF_y = -w + (-x*y)/sqrt_xy
    dF_z = 0

    dG_x = w + (-x*y)/sqrt_xy
    dG_y = 1 + (-2*y**2 - x**2)/sqrt_xy
    dG_z = 0

    dH_x = 0
    dH_y = 0
    dH_z = -1

    for i in range(0, 5):
        dtheta = theta - evt[i]
        component = np.exp(-dtheta**2/2*b**2) * (1 - dtheta**2/b**2)
        dH_x += component * (a[i]*y)/(x**2 + y**2)
        dH_y += component * (-a[i]*x)/(x**2 + y**2)

    return np.matrix([
        [dF_x, dF_y, dF_z],
        [dG_x, dG_y, dG_z],
        [dH_x, dH_y, dH_z]
    ])


def predict(X, P, Q, a, b, evt, w):
    '''
    X: state matrix         P: covariance matrix
    Q: process noise
    A: linearized ecg_model wrt to X
    F: linearized ecg_model wrt to noise

    a: constant             b: constant
    evt: constant           w: angular velocity

    Predict step:
    prior_X = f(x, w, k)
    prior_P = A_k * post_P * A_k^T + F_k * Q_k * F_k^T
    '''
    prior_X = ecg_model(X, a, b, evt, w)

    A = ecg_jacobian(X, a, b, evt, w)
    F = np.matrix([0, 0, 0], dtype="float").T
    prior_P = A*P*A.T + F*Q*F.T

    return (prior_X, prior_P)


def update(prior_X, prior_P, R, Y):
    '''
          Y: observation/measurements (amplitudes)
    prior_X: estimate to be updated with observation
     post_X: updated estimate of state with measurements
    prior_P: estimate of the covariance matrix
     post_P: updated estimate of covariance with measurements
          K: Kalman gain
          g: [0 0 1]X + v_k
          C: dg/dX
          G: dg/dv

    Update step:
         K = prior_P * C^T [C * prior_P * C^T + G]^(-1)
    post_X = prior_X + K[Y - g(x, 0, k)]
    post_P = prior_P - K * C * prior_P
    '''
    g = np.matrix([0, 0, 1], dtype="float") * prior_X.T + 0
    C = np.matrix([0, 0, 1], dtype="float")
    G = np.matrix([1], dtype="float")

    S = C * prior_P * C.T + G
    K = prior_P*C.T * S.I

    post_X = prior_X + K * (Y - g)
    post_P = prior_P - K * C * prior_P
    return (post_X, post_P)
