#!/usr/bin/env python3
import json
import re

import numpy as np
import numexpr as ne
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


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


def findpeak(ts, ys):
    peaks = find_peaks(ys, height=0, distance=70)
    peak_times = [ts[i] for i in peaks[0]]
    if len(peak_times) == 1:
        peak_period = np.max(ts) - peak_times[0]
    else:
        peak_period = np.mean(np.diff(peak_times))
    print(peaks)
    print(peak_times, peak_period)
    return peak_period


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

    alpha = 1 - np.sqrt(x**2 + y**2)
    dX[0] = (alpha*x) - (omega*y)
    dX[1] = (alpha*y) + (omega*x)
    dX[2] = -(z - z0) - sum(
        ai * omega * dthi * np.exp(-(dthi**2)/(2*bi**2))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    return dX


def discrete_ecg_model(X, a, b, evt, dt, omega=2*np.pi, z0=0):
    """Discrete version of ecg_model """
    x, y, z = X
    dX = np.zeros(3)
    # dX = np.zeros(18)
    theta = np.arctan2(y, x)
    dtheta = [(theta - ei) for ei in evt]

    alpha = 1 - np.sqrt(x**2 + y**2)
    dX[0] = (1+alpha*dt)*x - omega*dt*y
    dX[1] = (1+alpha*dt)*y + omega*dt*x
    dX[2] = -((dt-1)*z - dt*z0) - sum((
        ai * omega * dt * dthi * np.exp(-(dthi**2)/(2*bi**2)))
        for ai, bi, dthi in zip(a, b, dtheta)
    )
    # dX[3:18] = [*a, *b, *evt]
    return dX


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
