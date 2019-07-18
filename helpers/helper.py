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
        data = json.load()
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


def import_csv(file, timeframe):
    csvdata = np.genfromtxt(file, delimiter=',', skip_header=2)
    return csvdata[csvdata[:, 0] < timeframe]


def solve_ecg(a=None, b=None, evt=None, w=2*np.pi):
    def odefcn(T, Y, a, b, w, events):
        '''
        Function to solve ODE with scipy.integrate.solve_ivp()
        Details located here (p291):
        http://web.mit.edu/~gari/www/papers/ieeetbe50p289.pdf
        '''
        x, y, z = Y
        dy = np.zeros(3)

        theta = np.arctan2(y, x)
        alpha = 1.0 - np.sqrt(x**2 + y**2)
        dy[0] = alpha*x - w*y
        dy[1] = alpha*y + w*x

        dy[2] = -(z - 0)
        for i in range(0, 5):
            dtheta = theta - events[i]
            dy[2] -= a[i] * dtheta * np.exp(-(dtheta**2) / (2 * b[i]**2))

        return dy

    if a is None:
        a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
    if b is None:
        b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    if evt is None:
        evt = np.array([-np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2])

    tspan = np.array([-1.0, 1.0])
    y0 = np.array([-1.0, 0.0, 0.0])
    teval = np.linspace(0, 1, num=100)
    print('building...')
    sol = solve_ivp(fun=lambda t, y: odefcn(t, y, a, b, w, evt),
                    t_span=tspan, y0=y0, t_eval=teval)
    return sol
