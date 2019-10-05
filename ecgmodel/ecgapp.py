#!/usr/bin/env python3
import numexpr as ne

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.qt_compat import QtCore
from matplotlib.figure import Figure

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QDoubleValidator, QRegExpValidator
from PyQt5.QtWidgets import (QAction, QDesktopWidget, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLineEdit,
                             QMainWindow, QMenu, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget, qApp)

from ecgmodel import slider
from ecgmodel.helpers import helper


class ECGModel(QMainWindow):
    defaults = {
        'a': ["1.2", "-5.0", "30.0", "-7.5", "0.75"],
        'b': ["0.25", "0.1", "0.1", "0.1", "0.4"],
        'evt': ["-pi/3", "-pi/12", "0", "pi/12", "pi/2"],
        'omega': ["2*pi"]
    }

    def __init__(self, parent=None):
        super().__init__()

        self.initUI()

        # TODO: Find better way to calculate screen geometry

        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()
        print(ag, sg)

        self.setMinimumSize(1280, 550)
        # self.setGeometry(0, 0, 1500, 600)
        self.setWindowTitle('Synthetic ECG Waveform')
        self.show()

    def initUI(self):
        self.initMenu()

        self._mainwidget = QWidget()
        self._grid = QGridLayout()
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 0)

        self._mainwidget.setLayout(self._grid)
        self.setCentralWidget(self._mainwidget)

        self.init_ecgframe()
        self.init_formframe()

    def initMenu(self):
        menubar = self.menuBar()

        filemenu = menubar.addMenu('File')

        importmenu = QMenu('Import', self)
        importParam = QAction('Import Parameters', self)
        importParam.setShortcut('Ctrl+O')
        importParam.triggered.connect(self.import_params)
        importmenu.addAction(importParam)

        importSample = QAction('Import Sample', self)
        importSample.setShortcut('Ctrl+Shift+O')
        importSample.setStatusTip('Import sample ECG data')
        importSample.triggered.connect(self.import_sample)
        importmenu.addAction(importSample)

        filemenu.addMenu(importmenu)

        exportAction = QAction('Export', self)
        exportAction.setShortcut('Ctrl+S')
        exportAction.setStatusTip('Export current parameters')
        exportAction.triggered.connect(lambda: self.export_params())
        filemenu.addAction(exportAction)

        ekfAction = QAction('EKF', self)
        ekfAction.setStatusTip('Generate estimate from ECG sample')
        ekfAction.triggered.connect(self.buildEKF)
        filemenu.addAction(ekfAction)

        resetAction = QAction('Reset', self)
        resetAction.setShortcut('Ctrl+R')
        resetAction.setStatusTip('Reset current parameters')
        resetAction.triggered.connect(self.set_defaults)
        filemenu.addAction(resetAction)

        clearSample = QAction('Clear Sample', self)
        clearSample.setStatusTip('Remove ECG sample')
        clearSample.triggered.connect(lambda: self.removePlot('sample'))
        filemenu.addAction(clearSample)

        clearEstimate = QAction('Clear Estimate', self)
        clearEstimate.setStatusTip('Remove ECG estimate')
        clearEstimate.triggered.connect(lambda: self.removePlot())
        filemenu.addAction(clearEstimate)

        clearAll = QAction('Clear All', self)
        clearAll.setStatusTip('Clear all graphs')
        clearAll.triggered.connect(self.removeAll)
        filemenu.addAction(clearAll)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        filemenu.addAction(exitAction)

        aboutmenu = menubar.addMenu('About')

    def init_ecgframe(self):
        self.fig = Figure(figsize=(5, 3))
        self.fig.suptitle('Electrocardiogram Simulation')
        self.ax = self.fig.add_subplot(111, label='estimate')
        self.ax.set_xlabel('time (s)')
        self.ax.set_ylabel('mV')

        ecgframe = FigureCanvas(self.fig)
        self._grid.addWidget(ecgframe, 0, 0)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(ecgframe, self))

    def init_formframe(self):
        def make_entry(vdtr):
            entry = QLineEdit()
            entry.setFixedWidth(120)
            entry.setMaxLength(10)
            entry.setValidator(vdtr)
            return entry
        pi_re = QRegExp(r"[\d.+\-*/()pi\s]*")
        pi_validator = QRegExpValidator(pi_re)

        dbl_validator = QDoubleValidator(decimals=8)
        dbl_validator.setNotation(0)

        entries_a = QHBoxLayout()
        entries_b = QHBoxLayout()
        entries_evt = QHBoxLayout()
        for i in range(5):
            entry_a = make_entry(dbl_validator)
            entry_a.insert(str(ECGModel.defaults['a'][i]))

            entry_b = make_entry(dbl_validator)
            entry_b.insert(str(ECGModel.defaults['b'][i]))

            entry_evt = make_entry(pi_validator)
            entry_evt.insert(ECGModel.defaults['evt'][i])

            entries_a.addWidget(entry_a)
            entries_b.addWidget(entry_b)
            entries_evt.addWidget(entry_evt)

        entry_omega = make_entry(pi_validator)
        entry_omega.insert(ECGModel.defaults['omega'][0])

        # create a form
        self.formlayout = QFormLayout()
        self.formlayout.setVerticalSpacing(40)

        self.formlayout.addRow('a', entries_a)
        self.formlayout.addRow('b', entries_b)
        self.formlayout.addRow('theta', entries_evt)
        self.formlayout.addRow('omega', entry_omega)

        button = QPushButton('Build')
        button.setToolTip('Generate ECG with parameters')
        button.clicked.connect(self.buildParams)

        vbox = QVBoxLayout()
        vbox.addLayout(self.formlayout)
        vbox.addWidget(button)

        self.formframe = QGroupBox('Input Parameters:')
        self.formframe.setLayout(vbox)

        self._grid.addWidget(self.formframe, 0, 1)

    def get_entries(self):
        entries_a = self.formlayout.itemAt(0, 1)
        entries_b = self.formlayout.itemAt(1, 1)
        entries_evt = self.formlayout.itemAt(2, 1)

        a = [entries_a.itemAt(i).widget().text() for i in range(5)]
        b = [entries_b.itemAt(i).widget().text() for i in range(5)]
        evt = [entries_evt.itemAt(i).widget().text() for i in range(5)]
        omega = self.formlayout.itemAt(3, 1).widget().text()

        return (a, b, evt, omega)

    def set_entries(self, data):
        entries_a = self.formlayout.itemAt(0, 1)
        entries_b = self.formlayout.itemAt(1, 1)
        entries_evt = self.formlayout.itemAt(2, 1)
        entry_omega = self.formlayout.itemAt(3, 1).widget()

        for i in range(5):
            entry_a = entries_a.itemAt(i).widget()
            entry_a.clear()
            entry_a.insert(str(data['a'][i]))

            entry_b = entries_b.itemAt(i).widget()
            entry_b.clear()
            entry_b.insert(str(data['b'][i]))

            entry_evt = entries_evt.itemAt(i).widget()
            entry_evt.clear()
            entry_evt.insert(str(data['evt'][i]))

        entry_omega.clear()
        entry_omega.insert(data['omega'][0])

    def show_import_err(self, filename):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle('Import Error')
        msg.setText('Import Error: invalid file contains invalid data')
        msg.exec()

    def show_slider_timeframe(self, tmax=1):
        return slider.SliderDialog.getTimeFrame(self, tmax=tmax)

    def show_build_err(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('Build Error')
        msg.setText('Build Error: could not generate ECG, check for invalid parameters')
        msg.exec()

    def set_defaults(self):
        self.set_entries(ECGModel.defaults)

    def import_params(self):
        caption = 'Import Parameters'
        f_filter = 'JSON (*.json)'
        path = ''
        fileName = QFileDialog.getOpenFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        data = helper.import_json(fileName[0])
        if not data:
            self.show_import_err(fileName[0])
            return
        try:
            a, b, evt, omega = self.parseParams(data['a'], data['b'], data['evt'], data['omega'][0])
        except:
            self.show_import_err(fileName[0])
            return
        self.set_entries(data)
        self.buildECG(a, b, evt, omega)

    def export_params(self, filename='ecgdata.json'):
        caption = 'Export Parameters'
        path = 'ecgdata.json'
        f_filter = 'JSON (*.json)'
        fileName = QFileDialog.getSaveFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        a, b, evt, omega = self.get_entries()
        helper.export_json(fileName[0], a, b, evt, omega)

    def import_sample(self, filename, timeframe=1):
        caption = 'Import ECG Sample'
        f_filter = 'CSV (*.csv)'
        path = ''
        fileName = QFileDialog.getOpenFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        samples, tmax = helper.import_csv(fileName[0])
        if samples is None:
            helper.show_import_err(fileName[0])
        tframe, res = self.show_slider_timeframe(tmax=tmax)
        if not res:
            return

        data = helper.filter_timeframe(samples, tframe)

        self.removePlot('sample')
        self.ax.plot(data[0:, 0], data[0:, 1], 'r--', label='sample')
        self.redraw_axes()

    def removePlot(self, ln='estimate'):
        for l in self.ax.get_lines():
            if l.get_label() == ln:
                l.remove()
            self.redraw_axes()

    def removeAll(self):
        for l in self.ax.get_lines():
            l.remove()
        self.redraw_axes()

    def parseParams(self, a, b, evt, omega):
        arr_a = [float(i) for i in a]
        arr_b = [float(i) for i in b]
        arr_evt = [ne.evaluate(helper.pirepl(i), {'pi': helper.pi}) for i in evt]
        f_omega = ne.evaluate(helper.pirepl(omega), {'pi': helper.pi})

        return (arr_a, arr_b, arr_evt, f_omega)

    def redraw_axes(self):
        self.ax.relim()
        self.ax.autoscale_view()

        axe_hndl, axe_lbl = self.ax.get_legend_handles_labels()
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        if axe_lbl:
            self.ax.legend(handles=axe_hndl, labels=axe_lbl)
        self.ax.figure.canvas.draw()

    def buildParams(self):
        try:
            arr_a, arr_b, arr_evt, f_omega = self.parseParams(*self.get_entries())
        except:
            self.show_build_err()
            return
        self.buildECG(arr_a, arr_b, arr_evt, f_omega)

    def buildECG(self, a, b, evt, omega):
        self.removePlot()
        sol = helper.solve_ecg(a, b, evt, omega)
        self.ax.plot(sol.t, sol.y[2], 'b-', label='estimate')
        self.redraw_axes()

    def buildEKF(self):
        """
        Y (observations) = amplitudes taken from samples
        X_k (initial state) = [-1, 0, 0]
        P_k (covariance matrix) = [...]

        Q_k (measurement noise) = [...]
        R_k (process noise) = [1]

        a, b, evt (constants)
        omega (angular velocity) = approximation based on length of sample
        """
        sample = next((l for l in self.ax.get_lines() if l.get_label() == 'sample'), None)
        if not sample:
            print('no sample found!')
            return
        Y = sample.get_ydata()
        X = [-1, 0, 0]
        P = 10
        Q = 0
        R = 10
        a = [float(i) for i in ECGModel.defaults['a']]
        b = [float(i) for i in ECGModel.defaults['b']]
        evt = [ne.evaluate(helper.pirepl(i), {'pi': helper.pi}) for i in ECGModel.defaults['evt']]
        w = sample.get_xdata()[-1] * 2 * helper.pi

        res = helper.solve_ekf(Y, X, P, Q, R, a, b, evt, w)
        print(res)
