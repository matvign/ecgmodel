#!/usr/bin/env python3
import sympy as sp


sp.init_printing(use_latex="mathjax")
x, y, z = sp.symbols('x, y, z')
z0 = sp.symbols('z0')

a1, a2, a3, a4, a5 = sp.symbols('a1, a2, a3, a4, a5')
b1, b2, b3, b4, b5 = sp.symbols('b1, b2, b3, b4, b5')
e1, e2, e3, e4, e5 = sp.symbols('e1, e2, e3, e4, e5')
w = sp.symbols('w')
i = sp.symbols('i')

a_s = [a1, a2, a3, a4, a5]
b_s = [b1, b2, b3, b4, b5]
e_s = [e1, e2, e3, e4, e5]
dth_s = [(sp.atan2(y, x) - ei) for ei in e_s]

alpha = 1 - sp.sqrt(x**2 + y**2)
F = alpha * x - w * y
G = alpha * y + w * x
H = -(z - z0) - sum((ai * dthi * sp.exp(-(dthi**2)/(2*bi**2))) for ai, bi, dthi in zip(a_s, b_s, dth_s))
state = sp.Matrix([x, y, z])

xk, yk, zk, z0_k = 1, 1, 1, 0
ak = [1, 2, 3, 4, 5]
bk = [1, 2, 3, 4, 5]
ek = [1, 2, 3, 4, 5]
wk = 0

state = sp.Matrix([x, y, z])
m = sp.Matrix([F, G, H])
j = m.jacobian(state)

f = sp.lambdify([x, y, z, z0, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5,
                e1, e2, e3, e4, e5, w], j)

args = [xk, yk, zk, z0_k, *ak, *bk, *ek, wk]
res = f(*args)
print(res)
