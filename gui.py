import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.figure import Figure

import numpy as np
from scipy.integrate import solve_ivp

pi = np.pi


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


def initmenu(parent):
    '''
    Initialize menu components
    Menubar
      Filemenu
        Exit
      Aboutmenu
        About
    '''
    menubar = tk.Menu(parent)

    filemenu = tk.Menu(menubar)
    aboutmenu = tk.Menu(menubar)

    filemenu.add_command(label='Exit', command=parent.destroy)
    aboutmenu.add_command(label='About')

    menubar.add_cascade(label='File', menu=filemenu)
    menubar.add_cascade(label='About', menu=aboutmenu)
    parent.config(menu=menubar)


def initform(parent):
    vals = {
        "a": [1.2, -5.0, 30.0, -7.5, 0.75],
        "b": [0.25, 0.1, 0.1, 0.1, 0.4],
        "event": [-pi/3, -pi/12, 0, pi/12, pi/2]
    }

    a, b, events = [], [], []
    for i in range(0, 5):
        # create 5 labels and entries for a,b,event
        lbl_a = tk.Label(parent, text='a{}'.format(i+1))
        lbl_b = tk.Label(parent, text='b{}'.format(i+1))
        lbl_event = tk.Label(parent, text='theta{} (pi)'.format(i+1))

        entry_a = tk.Entry(parent, width=10)
        entry_a.insert(0, str(vals["a"][i]))

        entry_b = tk.Entry(parent, width=10)
        entry_b.insert(0, str(vals["b"][i]))

        entry_event = tk.Entry(parent, width=10)
        entry_event.insert(0, str(vals["event"][i]))

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
    lbl_w = tk.Label(parent, text='w (pi)').grid(row=8, column=0, sticky='e')
    entry_w = tk.Entry(parent, width=10)
    entry_w.insert(0, str(2*pi))
    entry_w.grid(row=8, column=1)


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


def app():
    def generatePlot():
        ecgplot.cla()
        sol = build_ecg()
        ecgplot.plot(sol.t, sol.y[2], 'b-')
        ecgcanvas.draw()

    root = tk.Tk()

    # remove tearoff from menus
    root.option_add('*tearOff', 0)
    root.wm_title("Embedding in Tk")
    root.geometry('900x450')

    initmenu(root)

    # create frames and initialise their grids
    content = tk.Frame(root)
    ecgframe = tk.Frame(content, bd=4, relief='sunken', height=400, width=500)
    formframe = tk.Frame(content)
    buttonFrame = tk.Frame(content)

    content.grid(sticky='nsew')
    ecgframe.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=(10, 0))
    formframe.grid(row=0, column=1, columnspan=1, sticky='nsew', pady=(10, 0))
    buttonFrame.grid(row=1, column=1, columnspan=1, sticky='nsew')

    initform(formframe)
    button = tk.Button(formframe, text='Build', command=generatePlot, pady=8)
    button.grid(row=12, column=1, columnspan=2, pady=20)

    # build ecg and display it in the ecgframe
    fig = Figure(figsize=(5, 4), dpi=100)

    ecgcanvas = FigureCanvasTkAgg(fig, ecgframe)
    ecgcanvas.get_tk_widget().grid()

    ecgplot = fig.add_subplot(111)
    generatePlot()

    # create a toolbar for the figure in the ecgframe
    toolbarFrame = tk.Frame(root)
    toolbarFrame.grid(row=1, column=0, columnspan=12, sticky='w')
    toolbar = NavigationToolbar2Tk(ecgcanvas, toolbarFrame)
    toolbar.update()

    root.mainloop()


if __name__ == "__main__":
    app()