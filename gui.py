import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np
from scipy.integrate import solve_ivp


def odefcn(T, Y, a, b, w, events):
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


def build_ecg():
    pass


def initmenu(parent):
    menubar = tk.Menu(parent)
    filemenu = tk.Menu(menubar)
    filemenu.add_command(label='Exit', command=parent.quit)
    menubar.add_cascade(label='File', menu=filemenu)

    aboutmenu = tk.Menu(menubar)
    aboutmenu.add_command(label='About')
    menubar.add_cascade(label='About', menu=aboutmenu)
    parent.config(menu=menubar)


def initform(parent, lst=None):
    def pairwise(it):
        it = iter(it)
        while True:
            yield next(it), next(it)

    a, b, events = [], [], []
    forms = []
    for i in range(0, 5):
        lbl_a = tk.Label(parent, text='a{}'.format(i+1))
        lbl_b = tk.Label(parent, text='b{}'.format(i+1))
        lbl_event = tk.Label(parent, text='theta{} (pi)'.format(i+1))

        a.extend([lbl_a, tk.Entry(parent, width=10)])
        b.extend([lbl_b, tk.Entry(parent, width=10)])
        events.extend([lbl_event, tk.Entry(parent, width=10)])

    forms.extend(a+b+events)
    for count, (lbl, entry) in enumerate(pairwise(forms)):
        col = 2 if count % 2 else 0
        ro = int(count/2)
        lbl.grid(row=ro, column=col, sticky='e')
        entry.grid(row=ro, column=col+1)

    lbl_w = tk.Label(parent, text='w (pi)').grid(row=8, column=0, sticky='e')
    entry_w = tk.Entry(parent, width=10).grid(row=8, column=1)
    button = tk.Button(parent, text='Build', command=build_ecg, pady=8)
    button.grid(row=12, column=1, columnspan=2, pady=20)


root = tk.Tk()
# remove tearoff from all menus
root.option_add('*tearOff', 0)
root.wm_title("Embedding in Tk")
root.geometry('900x450')
initmenu(root)

pi = np.pi
events = np.array([-pi/3, -pi/12, 0, pi/12, pi/2])
a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
w = 2 * pi
tspan = np.array([0.0, 1.0])
y0 = np.array([-1.0, 0.0, 0.0])
sol = solve_ivp(fun=lambda t, y: odefcn(t, y, a, b, w, events), t_span=tspan, y0=y0)

fig = Figure(figsize=(5, 4), dpi=100)
fig.add_subplot(111).plot(sol.t, sol.y[2], 'b-')

content = tk.Frame(root)
ecgframe = tk.Frame(content, bd=4, relief='sunken', height=400, width=500)
formframe = tk.Frame(content)
buttonFrame = tk.Frame(content)

content.grid(sticky='nsew')
ecgframe.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=(10, 0))
formframe.grid(row=0, column=1, columnspan=1, sticky='nsew', pady=(10, 0))
buttonFrame.grid(row=1, column=1, columnspan=1, sticky='nsew')

ecgcanvas = FigureCanvasTkAgg(fig, ecgframe)
ecgcanvas.get_tk_widget().grid()
ecgcanvas.draw()

toolbarFrame = tk.Frame(root)
toolbarFrame.grid(row=1, column=0, columnspan=12, sticky='w')
toolbar = NavigationToolbar2Tk(ecgcanvas, toolbarFrame)
toolbar.update()
initform(formframe)

root.mainloop()