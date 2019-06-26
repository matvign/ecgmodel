#!/usr/bin/env python3
import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.figure import Figure

import numpy as np
from scipy.integrate import solve_ivp

pi = np.pi


class MainApplication(tk.Frame):
    def __init__(self):
        super().__init__()
        self.defaults = {
            'a': [1.2, -5.0, 30.0, -7.5, 0.75],
            'b': [0.25, 0.1, 0.1, 0.1, 0.4],
            'event': [-pi/3, -pi/12, 0, pi/12, pi/2]
        }
        self.initUI()

    def initUI(self):
        self.master.title('Embedding in Tk')
        self.master.geometry('900x450')
        self.master.option_add('*tearOff', 0)

        self.content = tk.Frame(self.master)
        self.ecgframe = tk.Frame(self.content, bd=4, relief='sunken', height=400, width=500)
        self.formframe = tk.Frame(self.content)
        self.buttonframe = tk.Frame(self.content)

        self.content.grid(sticky='nsew')
        self.ecgframe.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=(10, 0))
        self.formframe.grid(row=0, column=1, columnspan=1, sticky='nsew', pady=(10, 0))
        self.buttonframe.grid(row=1, column=1, columnspan=1, sticky='nsew')

        self.initmenu()
        self.initform()

        self.buildbtn = tk.Button(self.formframe, text='Build', command=self.build_ecg, pady=8)
        self.buildbtn.grid(row=12, column=1, columnspan=2, pady=20)

        figure = Figure(figsize=(5, 4), dpi=100)
        self.fig = figure.add_subplot(111)
        self.ecgcanvas = FigureCanvasTkAgg(figure, self.ecgframe)
        self.ecgcanvas.get_tk_widget().grid()

        toolbarframe = tk.Frame(self.content)
        toolbarframe.grid(row=1, column=0, columnspan=12, sticky='w')
        self.toolbar = NavigationToolbar2Tk(self.ecgcanvas, toolbarframe)
        self.toolbar.update()
        # self.build_ecg()

    def initmenu(self):
        self.menubar = tk.Menu(self.master)
        filemenu = tk.Menu(self.menubar)
        aboutmenu = tk.Menu(self.menubar)

        filemenu.add_command(label='Exit', command=self.master.destroy)
        aboutmenu.add_command(label='About')

        self.menubar.add_cascade(label='File', menu=filemenu)
        self.menubar.add_cascade(label='About', menu=aboutmenu)
        self.master.config(menu=self.menubar)

    def initform(self):
        vcmd = (self.register(self.onValidate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        a, b, events = [], [], []
        for i in range(0, 5):
            # create 5 labels and entries for a,b,event
            lbl_a = tk.Label(self.formframe, text='a{}'.format(i+1))
            lbl_b = tk.Label(self.formframe, text='b{}'.format(i+1))
            lbl_event = tk.Label(self.formframe, text='theta{} (pi)'.format(i+1))

            entry_a = tk.Entry(self.formframe, validate='key',
                               validatecommand=vcmd, width=10)
            entry_a.insert(0, self.defaults['a'][i])
            entry_a.label = 'a'
            entry_a.number = i

            entry_b = tk.Entry(self.formframe, validate='key',
                               validatecommand=vcmd, width=10)
            entry_b.insert(0, self.defaults['b'][i])
            entry_b.label = 'b'
            entry_b.number = i

            entry_event = tk.Entry(self.formframe, validate='key',
                                   validatecommand=vcmd, width=10)
            entry_event.insert(0, self.defaults['event'][i])
            entry_event.label = 'event'
            entry_event.number = i

            a.append((lbl_a, entry_a))
            b.append((lbl_b, entry_b))
            events.append((lbl_event, entry_event))

            for count, (lbl, entry) in enumerate(a+b+events):
                # create right aligned labels and entries
                col = 2 if count % 2 else 0
                ro = int(count/2)
                lbl.grid(row=ro, column=col, sticky='e')
                entry.grid(row=ro, column=col+1)

        # create label+entry for omega, the angular velocity
        lbl_w = tk.Label(self.formframe, text='w (pi)').grid(row=8, column=0, sticky='e')
        entry_w = tk.Entry(self.formframe, width=10)
        entry_w.insert(0, str(2*pi))
        entry_w.grid(row=8, column=1)

    def onValidate(self, d, i, P, s, S, v, V, W):
        # valid percent substitutions (from the Tk entry man page)
        # note: you only have to register the ones you need
        #
        # %d = Type of action (1=insert, 0=delete, -1 for others)
        # %i = index of char string to be inserted/deleted, or -1
        # %P = value of the entry if the edit is allowed
        # %s = value of entry prior to editing
        # %S = the text string being inserted or deleted, if any
        # %v = the type of validation that is currently set
        # %V = the type of validation that triggered the callback
        #      (key, focusin, focusout, forced)
        # %W = the tk name of the widget

        # dont allow anything that can't be parsed as a float
        try:
            float(P)
            return True
        except ValueError:
            if not P:
                return True
            return False

    def build_ecg(self):
        inputs = []
        for e in self.formframe.winfo_children():
            if isinstance(e, tk.Entry):
                s = e.get().strip()
                if not s:
                    e.insert(0, self.defaults[e.label][e.number])
                i = float(e.get())
                inputs.append(i)
        a, b, evt = [], [], []
        for (ai, bi, ei) in triway(inputs):
            a.append(ai)
            b.append(bi)
            evt.append(ei)

        self.fig.cla()
        # sol = build_ecg()
        sol = build_ecg(np.array(a), np.array(b), np.array(evt))
        self.fig.plot(sol.t, sol.y[2], 'b-')
        self.ecgcanvas.draw()


def triway(lst):
    it = iter(lst)
    while True:
        try:
            yield next(it), next(it), next(it)
        except StopIteration:
            return


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


def build_ecg(a=None, b=None, evt=None, w=2*pi):
    if a is None:
        a = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
    if b is None:
        b = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
    if evt is None:
        evt = np.array([-pi/3, -pi/12, 0, pi/12, pi/2])

    tspan = np.array([0.0, 1.0])
    y0 = np.array([-1.0, 0.0, 0.0])
    print('building...')
    sol = solve_ivp(fun=lambda t, y: odefcn(t, y, a, b, w, evt), t_span=tspan, y0=y0)
    return sol


def main():
    root = tk.Tk()
    app = MainApplication()
    root.mainloop()


if __name__ == "__main__":
    main()