#!/usr/bin/env python3
import json
import re

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import numexpr as ne
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from helpers import helper

pi = np.pi
# pattern to allow only numbers, math operators +-*/() and pi
mathops = re.compile(r"^[0-9.+\-*/()pi\b]*$")


class ECGModel(ttk.Frame):
    def __init__(self):
        super().__init__()
        self.master.title('Embedding in Tk')
        self.master.geometry('950x450')
        self.master.option_add('*tearOff', 0)

        self.preset = {
            'default': {
                'a': [1.2, -5.0, 30.0, -7.5, 0.75],
                'b': [0.25, 0.1, 0.1, 0.1, 0.4],
                'event': ["-pi/3", "-pi/12", "0", "pi/12", "pi/2"],
                'omega': ["2*pi"]
            }
        }
        self.initStyles()
        self.initUI()

    def initStyles(self):
        self.style = ttk.Style()
        self.style.theme_use("default")
        self.style.configure("TFrame", bg="#D9D9D9")

    def initUI(self):
        self.pack(fill=tk.BOTH, expand=True)

        # make grids under master expand to window (except for row 1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.initmenu()

        # initialize ecgframe into master grid col=0, row=0
        # have it expand in all directions
        # make grids under ecgframe expand to the size of this frame
        # when master is resized, this frame and its contents will also resize
        self.ecgframe = ttk.Frame(self)
        self.ecgframe.grid(column=0, row=0, sticky='nsew', padx=10, pady=(10, 0))
        self.ecgframe.columnconfigure(0, weight=1)
        self.ecgframe.rowconfigure(0, weight=1)

        # initialize canvas into ecgframe grid col=0, row=0
        # have it expand in all directions
        figure = Figure(figsize=(5, 4), dpi=100)
        self.fig = figure.add_subplot(111)
        self.ecgcanvas = FigureCanvasTkAgg(figure, self.ecgframe)
        self.ecgcanvas.get_tk_widget().grid(column=0, row=0, sticky='nsew')

        # initialize formframe into master grid col=1, row=0
        # have it expand in all directions
        self.formframe = ttk.Frame(self)
        self.formframe.grid(column=1, row=0, sticky='ns', pady=(10, 0))
        for n in range(0, 10):
            self.formframe.rowconfigure(n, weight=1)
        self.initform()

        toolbarframe = ttk.Frame(self)
        toolbarframe.grid(column=0, row=1, padx=(10, 0), sticky='nw')

        # create a toolbar that tracks the canvas and belongs to toolbarframe
        self.toolbar = NavigationToolbar2Tk(self.ecgcanvas, toolbarframe)
        self.toolbar.update()

    def initmenu(self):
        self.menubar = tk.Menu(self.master)
        filemenu = tk.Menu(self.menubar)
        aboutmenu = tk.Menu(self.menubar)

        filemenu.add_command(label='Import', command=self.import_file)
        filemenu.add_command(label='Export', command=self.export_file)
        filemenu.add_command(label='Reset', command=self.reset_form)
        filemenu.add_command(label='Exit', command=self.master.destroy)

        aboutmenu.add_command(label='About')

        self.menubar.add_cascade(label='File', menu=filemenu)
        self.menubar.add_cascade(label='About', menu=aboutmenu)
        self.master.config(menu=self.menubar)

    def initform(self):
        vcmdAB = (self.register(self.validateAB),
            '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        vcmdEvt = (self.register(self.validateEvent),
            '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        a, b, events = [], [], []
        for i in range(0, 5):
            # create 5 labels and entries for a,b,event
            lbl_a = ttk.Label(self.formframe, text='a{}'.format(i+1))
            entry_a = ttk.Entry(self.formframe, validate='key',
                               validatecommand=vcmdAB, width=10)
            entry_a.insert(0, self.preset['default']['a'][i])
            entry_a.label = 'a'
            entry_a.number = i

            lbl_b = ttk.Label(self.formframe, text='b{}'.format(i+1))
            entry_b = ttk.Entry(self.formframe, validate='key',
                               validatecommand=vcmdAB, width=10)
            entry_b.insert(0, self.preset['default']['b'][i])
            entry_b.label = 'b'
            entry_b.number = i

            lbl_event = ttk.Label(self.formframe, text='theta{} (pi)'.format(i+1))
            entry_event = ttk.Entry(self.formframe, validate='key',
                                   validatecommand=vcmdEvt, width=10)
            entry_event.insert(0, self.preset['default']['event'][i])
            entry_event.label = 'event'
            entry_event.number = i

            a.append((lbl_a, entry_a))
            b.append((lbl_b, entry_b))
            events.append((lbl_event, entry_event))

        for count, (lbl, entry) in enumerate(a+b+events):
            # create right aligned labels and entries
            col = 2 if count % 2 else 0
            ro = int(count/2)
            lbl.grid(column=col, row=ro, sticky='w', pady=3)
            entry.grid(column=col+1, row=ro, pady=3)

        # create label+entry for omega, the angular velocity
        lbl_w = ttk.Label(self.formframe, text='w (pi)').grid(row=8, column=0, sticky='w')
        entry_w = ttk.Entry(self.formframe, width=10)
        entry_w.label = 'omega'
        entry_w.number = 0
        entry_w.insert(0, "2*pi")
        entry_w.grid(row=8, column=1, pady=3)

        self.buildbtn = ttk.Button(self.formframe, text='Build',
            command=self.build_ecg)
        self.buildbtn.grid(column=1, row=9, sticky='s', columnspan=2, pady=20)

    def load_preset(self, data):
        for e in self.formframe.winfo_children():
            if not isinstance(e, tk.Entry):
                continue

            label, number = e.label, e.number
            e.delete(0, tk.END)
            e.insert(0, data[label][number])

    def reset_form(self):
        self.load_preset(self.preset['default'])
        self.build_ecg()

    def get_json(self):
        err1 = "Cannot export file with empty value"
        err2 = "Cannot export file with invalid mathematical expression"

        data = {'a': [], 'b': [], 'event': [], 'omega': []}
        for e in self.formframe.winfo_children():
            if not isinstance(e, tk.Entry):
                continue

            label, value = e.label, e.get().replace(' ', '')
            if not value:
                tk.messagebox.showerror("Error", err1)
                return
            if label == 'a' or label == 'b':
                i = float(value)
                data[label].append(i)
            else:
                s = helper.pirepl(value)
                try:
                    i = ne.evaluate(s)
                    data[label].append(s)
                except:
                    tk.messagebox.showerror("Error", err2)
        return data

    def import_file(self):
        '''
        import json data to use
        load this information into the forms
        '''
        ftypes = [('JSON files', '*.json')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()
        if fl:
            with open(fl) as f:
                data = json.load(f)
            self.load_preset(data)
            self.build_ecg()

    def export_file(self, filename='ecgdata.json'):
        '''
        export json data to use
        '''
        print('Exporting data...')
        with open(filename, "w") as f:
            json.dump(self.get_json(), f, indent=4)

    def build_ecg(self):
        errmsg = "Could not interpret mathematical expression"
        a, b, evt = [], [], []

        for e in self.formframe.winfo_children():
            if not isinstance(e, tk.Entry):
                continue

            label, number, value = e.label, e.number, e.get().replace(' ', '')
            if not value:
                e.insert(0, self.preset['default'][label][number])

            if label == 'a' or label == 'b':
                i = float(value)
                a.append(i) if label == 'a' else b.append(i)

            else:
                s = helper.pirepl(value)
                try:
                    i = ne.evaluate(s)
                except:
                    tk.messagebox.showerror("Error", errmsg)
                    return
                omega = i if label == 'omega' else evt.append(i)

        self.fig.cla()
        sol = build_ecg(np.array(a), np.array(b), np.array(evt), omega)
        self.fig.plot(sol.t, sol.y[2], 'b-')
        self.ecgcanvas.draw()

    def validateEvent(self, d, i, P, s, S, v, V, W):
        # only allow numbers, operators -, +, *, /, round brackets
        # and the keyword pi
        if (re.match(mathops, P)):
            return True
        return False

    def validateAB(self, d, i, P, s, S, v, V, W):
        # dont allow anything that can't be parsed as a float
        try:
            float(P)
            return True
        except ValueError:
            if not P:
                return True
            return False


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

    tspan = np.array([-1.0, 1.0])
    y0 = np.array([-1.0, 0.0, 0.0])
    teval = np.linspace(0, 1, num=100)
    print('building...')
    sol = solve_ivp(fun=lambda t, y: odefcn(t, y, a, b, w, evt),
                    t_span=tspan, y0=y0, t_eval=teval)
    return sol


def main():
    root = tk.Tk()

    app = ECGModel()
    root.mainloop()