#!/usr/bin/env python3
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

from ecgmodel import slider, ekf_form
from ecgmodel.helpers import helper, parameter_fit


class ECGModel(QMainWindow):
    defaults = {
        "a": ["12", "-50", "300", "-75", "7.5"],
        "b": ["0.25", "0.1", "0.1", "0.1", "0.4"],
        "evt": ["-pi/3", "-pi/12", "0", "pi/12", "pi/2"],
        "omega": ["2*pi"]
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
        self.setWindowTitle("Synthetic ECG Waveform")
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
        filemenu = menubar.addMenu("File")

        # import menu
        importParam = QAction("Import Parameters", self)
        importParam.setShortcut("Ctrl+O")
        importParam.triggered.connect(self.import_params)

        importSample = QAction("Import Sample", self)
        importSample.setShortcut("Ctrl+Shift+O")
        importSample.setStatusTip("Import sample ECG data")
        importSample.triggered.connect(self.import_sample)

        importmenu = QMenu("Import", self)
        importmenu.addAction(importParam)
        importmenu.addAction(importSample)

        # export menu
        exportParam = QAction("Export Parameters", self)
        exportParam.setShortcut("Ctrl+Shift+S")
        exportParam.setStatusTip("Export current parameters")
        exportParam.triggered.connect(lambda: self.export_params())

        exportCSV = QAction("Export CSV", self)
        exportCSV.setShortcut("Ctrl+S")
        exportCSV.setStatusTip("Export generated ECG")
        exportCSV.triggered.connect(lambda: self.export_csv())

        exportmenu = QMenu("Export", self)
        exportmenu.addAction(exportParam)
        exportmenu.addAction(exportCSV)

        # clear menu
        clearEstimate = QAction("Clear Estimate", self)
        clearEstimate.setStatusTip("Remove ECG estimate")
        clearEstimate.triggered.connect(lambda: self.removePlot())

        clearSample = QAction("Clear Sample", self)
        clearSample.setStatusTip("Remove ECG sample")
        clearSample.triggered.connect(lambda: self.removePlot("sample"))

        # clearEKF = QAction("Clear EKF", self)
        # clearEKF.setStatusTip("Remove EKF plot")
        # clearEKF.triggered.connect(lambda: self.removePlot("paramfit"))

        clearAll = QAction("Clear All", self)
        clearAll.setStatusTip("Clear all graphs")
        clearAll.triggered.connect(self.removeAll)

        clearmenu = QMenu("Clear", self)
        clearmenu.addAction(clearEstimate)
        clearmenu.addAction(clearSample)
        # clearmenu.addAction(clearEKF)
        clearmenu.addAction(clearAll)

        paramfitAction = QAction("Parameter Fit Sample", self)
        paramfitAction.setStatusTip("Parameter Fit Sample ECG")
        # paramfitAction.triggered.connect(self.parameter_fit)
        paramfitAction.triggered.connect(self.show_ekf_form)

        resetAction = QAction("Reset", self)
        resetAction.setShortcut("Ctrl+R")
        resetAction.setStatusTip("Reset current parameters")
        resetAction.triggered.connect(self.set_defaults)

        # peakAction = QAction("Peak", self)
        # peakAction.setStatusTip("Find Peaks")
        # peakAction.triggered.connect(self.peakfind)

        exitAction = QAction("Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(qApp.quit)

        filemenu.addMenu(importmenu)
        filemenu.addMenu(exportmenu)
        filemenu.addMenu(clearmenu)
        filemenu.addAction(paramfitAction)
        filemenu.addAction(resetAction)
        # filemenu.addAction(peakAction)
        filemenu.addAction(exitAction)
        aboutmenu = menubar.addMenu("About")

    def init_ecgframe(self):
        self.fig = Figure(figsize=(5, 3))
        self.fig.suptitle("Electrocardiogram Simulation")
        self.ax = self.fig.add_subplot(111, label="estimate")
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("mV")

        ecgframe = FigureCanvas(self.fig)
        self._grid.addWidget(ecgframe, 0, 0)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(ecgframe, self))

    def init_formframe(self):
        def make_entry(val, vdtr):
            entry = QLineEdit(val)
            entry.setFixedWidth(120)
            entry.setMaxLength(10)
            entry.setValidator(vdtr)
            return entry

        pi_validator = QRegExpValidator(QRegExp(r"[\d.+\-*/()pi\s]*"))
        dbl_validator = QDoubleValidator(decimals=8)
        dbl_validator.setNotation(0)

        entries_a = QHBoxLayout()
        entries_b = QHBoxLayout()
        entries_evt = QHBoxLayout()
        for i in range(5):
            entry_a = make_entry(str(ECGModel.defaults["a"][i]), dbl_validator)
            entry_b = make_entry(str(ECGModel.defaults["b"][i]), dbl_validator)
            entry_evt = make_entry(ECGModel.defaults["evt"][i], pi_validator)

            entries_a.addWidget(entry_a)
            entries_b.addWidget(entry_b)
            entries_evt.addWidget(entry_evt)

        entry_omega = make_entry(ECGModel.defaults["omega"][0], pi_validator)

        t_entry = QLineEdit(str(1))
        t_entry.setFixedWidth(50)
        t_entry.setMaxLength(5)
        t_validator = QRegExpValidator(QRegExp(r"\d*"))
        t_entry.setValidator(t_validator)

        # create a form
        self.formlayout = QFormLayout()
        self.formlayout.setVerticalSpacing(40)

        self.formlayout.addRow("a", entries_a)
        self.formlayout.addRow("b", entries_b)
        self.formlayout.addRow("theta", entries_evt)
        self.formlayout.addRow("omega", entry_omega)
        self.formlayout.addRow("t (s)", t_entry)

        button = QPushButton("Build")
        button.setToolTip("Generate ECG with parameters")
        button.clicked.connect(self.buildParams)

        vbox = QVBoxLayout()
        vbox.addLayout(self.formlayout)
        vbox.addWidget(button)

        self.formframe = QGroupBox("Input Parameters:")
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
        tf = int(self.formlayout.itemAt(4, 1).widget().text())

        return (a, b, evt, omega, tf)

    def set_entries(self, data):
        entries_a = self.formlayout.itemAt(0, 1)
        entries_b = self.formlayout.itemAt(1, 1)
        entries_evt = self.formlayout.itemAt(2, 1)
        entry_omega = self.formlayout.itemAt(3, 1).widget()

        for i in range(5):
            entry_a = entries_a.itemAt(i).widget()
            entry_a.clear()
            entry_a.insert(str(data["a"][i]))

            entry_b = entries_b.itemAt(i).widget()
            entry_b.clear()
            entry_b.insert(str(data["b"][i]))

            entry_evt = entries_evt.itemAt(i).widget()
            entry_evt.clear()
            entry_evt.insert(str(data["evt"][i]))

        entry_omega.clear()
        entry_omega.insert(str(data["omega"][0]))

    def set_defaults(self):
        self.set_entries(ECGModel.defaults)

    def show_slider_timeframe(self, tmax=1):
        return slider.SliderDialog.getTimeFrame(self, tmax=tmax)

    def show_ekf_form(self):
        sample = next((l for l in self.ax.get_lines() if l.get_label() == "sample"), None)
        if not sample:
            self.show_warning("Warning", "No sample to estimate paramters")
            return
        opts, res = ekf_form.KalmanFilterForm.get_ekf_options(self)
        print(opts, res)
        if not res:
            return
        self.parameter_fit(sample, opts)

    def show_warning(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def show_information(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec()

    def import_params(self):
        caption = "Import Parameters"
        f_filter = "JSON (*.json)"
        path = ""
        fileName = QFileDialog.getOpenFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        data = helper.import_json(fileName[0])
        try:
            if not data:
                self.show_warning("Import Error", "Import Error: file contains invalid parameters")
                return
            a, b, evt, omega = self.parseParams(data["a"], data["b"], data["evt"], data["omega"][0])
            tf = int(self.formlayout.itemAt(4, 1).widget().text())
        except:
            self.show_warning("Import Error", "Import Error: file contains invalid parameters")
            return
        self.set_entries(data)
        self.buildECG(a, b, evt, omega, tf)

    def import_sample(self, filename, timeframe=1):
        caption = "Import ECG Sample"
        f_filter = "CSV (*.csv)"
        path = ""
        fileName = QFileDialog.getOpenFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        samples = helper.import_csv(fileName[0])
        tmax = samples[:, 0][-1]

        if samples is None:
            self.show_warning("Import Error", "Import Error: sample contains invalid data")
        tframe, res = self.show_slider_timeframe(tmax=tmax)
        if not res:
            return

        data = helper.filter_timeframe(samples, tframe)
        self.removePlot("sample")
        self.ax.plot(data[0:, 0], data[0:, 1], "b-", label="sample")
        self.redraw_axes()

    def export_params(self, filename="ecgdata.json"):
        caption = "Export Parameters"
        path = filename
        f_filter = "JSON (*.json)"
        fileName = QFileDialog.getSaveFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        a, b, evt, omega, _ = self.get_entries()
        helper.export_json(fileName[0], a, b, evt, omega)

    def export_csv(self, filename="ecg.csv"):
        ecg = next((l for l in self.ax.get_lines() if l.get_label() == "estimate"), None)
        if not ecg:
            self.show_information("Export CSV", "No sample to estimate paramters")
            return
        caption = "Export Parameters"
        path = filename
        f_filter = "CSV (*.csv)"
        fileName = QFileDialog.getSaveFileName(self, caption, path, f_filter)
        if not fileName[0]:
            return
        helper.export_csv(fileName[0], ecg)

    def removePlot(self, ln="estimate"):
        for l in self.ax.get_lines():
            if l.get_label() == ln:
                l.remove()
            self.redraw_axes()

    def removeAll(self):
        for l in self.ax.get_lines():
            l.remove()
        self.redraw_axes()

    def parseParams(self, a, b, evt, omega):
        a_ = [float(i) for i in a]
        b_ = [float(i) for i in b]
        evt_ = [helper.convert_pi(i) for i in evt]
        omega_ = helper.convert_pi(omega)
        return (a_, b_, evt_, omega_)

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
            a_, b_, evt_, omega_, t = self.get_entries()
            a, b, evt, omega = self.parseParams(a_, b_, evt_, omega_)
            t = int(t)
        except:
            self.show_information("Build Error", "Build Error: could not generate ECG, check for invalid parameters")
            return
        self.buildECG(a, b, evt, omega, t)

    def buildECG(self, a, b, evt, omega, tf):
        self.removePlot()
        sol = helper.solve_ecg(a, b, evt, omega, tf)
        self.ax.plot(sol.t, sol.y[2], "k--", label="estimate")
        self.redraw_axes()

    def parameter_fit(self, sample, opts):
        if not sample:
            self.show_warning("Warning", "No sample to estimate paramters")
            return
        ts = sample.get_xdata()
        ys = sample.get_ydata()

        try:
            a_, b_, e_, omega_, _ = self.get_entries()
            a, b, e, omega = self.parseParams(a_, b_, e_, omega_)
            # rr = helper.findpeak(ts, ys)
            # omega = 2 * helper.pi / rr
        except:
            self.show_information("Build Error", "Build Error: could not generate ECG, check for invalid parameters")
            return

        res = parameter_fit.parameter_est(ts, ys, a, b, e, omega, opts)
        # print(res[2])
        data = {
            "a": res[2][0:5],
            "b": res[2][5:10],
            "evt": res[2][10:15],
            "omega": [res[2][15]]
        }
        # self.removePlot("paramfit")
        # self.ax.plot(res[0], res[1], 'r--', label='paramfit')
        self.set_entries(data)
        self.show_information("Parameter Estimate", "Finished Parameter Estimation")
        self.redraw_axes()
