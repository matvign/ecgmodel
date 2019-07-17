#!/usr/bin/env python3
import numexpr as ne

from numpy import array as nparr
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QDoubleValidator, QRegExpValidator
from PyQt5.QtWidgets import (qApp, QWidget, QDesktopWidget, QMainWindow,
    QFileDialog, QMessageBox, QLineEdit, QPushButton, QAction)
from PyQt5.QtWidgets import (QGridLayout, QHBoxLayout, QVBoxLayout,
    QGroupBox, QFormLayout)

from helpers import helper


class ECGModel(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()

        self.initUI()

        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()
        print(ag, sg)
        self.setGeometry(0, 0, 1500, 600)
        self.setWindowTitle('Synthetic ECG Waveform')
        self.show()

    def initUI(self):
        self.initParams()
        self.initMenu()

        self._mainwidget = QWidget()
        self._grid = QGridLayout()
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 0)

        self._mainwidget.setLayout(self._grid)
        self.setCentralWidget(self._mainwidget)

        self.init_ecgframe()
        self.init_formframe()

    def initParams(self):
        self.exprdict = {'pi': helper.pi}
        self.preset = {
            'defaults': {
                'a': ["1.2", "-5.0", "30.0", "-7.5", "0.75"],
                'b': ["0.25", "0.1", "0.1", "0.1", "0.4"],
                'evt': ["-pi/3", "-pi/12", "0", "pi/12", "pi/2"],
                'omega': ["2*pi"]
            }
        }

    def initMenu(self):
        menubar = self.menuBar()

        filemenu = menubar.addMenu('File')

        importAction = QAction('Import', self)
        importAction.setShortcut('Ctrl+O')
        importAction.setStatusTip('Import file for parameters')
        importAction.triggered.connect(self.importParams)

        exportAction = QAction('Export', self)
        exportAction.setShortcut('Ctrl+S')
        exportAction.setStatusTip('Export current parameters')
        exportAction.triggered.connect(lambda: self.exportParams())

        resetAction = QAction('Reset', self)
        resetAction.setShortcut('Ctrl+R')
        resetAction.setStatusTip('Reset current parameters')
        resetAction.triggered.connect(self.defaultParams)

        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        filemenu.addAction(importAction)
        filemenu.addAction(exportAction)
        filemenu.addAction(resetAction)
        filemenu.addAction(exitAction)

        aboutmenu = menubar.addMenu('About')

    def init_ecgframe(self):
        ecgframe = FigureCanvas(Figure(figsize=(5, 3)))
        self._grid.addWidget(ecgframe, 0, 0)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(ecgframe, self))
        self.ecgplot = ecgframe.figure.subplots()

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

        defaults = self.preset['defaults']

        entries_a = QHBoxLayout()
        entries_b = QHBoxLayout()
        entries_evt = QHBoxLayout()
        for i in range(5):
            entry_a = make_entry(dbl_validator)
            entry_a.insert(str(defaults['a'][i]))

            entry_b = make_entry(dbl_validator)
            entry_b.insert(str(defaults['b'][i]))

            entry_evt = make_entry(pi_validator)
            entry_evt.insert(defaults['evt'][i])

            entries_a.addWidget(entry_a)
            entries_b.addWidget(entry_b)
            entries_evt.addWidget(entry_evt)

        entry_omega = make_entry(pi_validator)
        entry_omega.insert(defaults['omega'][0])

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

        self.formframe = QGroupBox('Input Parameters')
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

    def show_build_err(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('Build Error')
        msg.setText('Build Error: could not generate ECG, check for invalid parameters')
        msg.exec()

    def defaultParams(self):
        data = self.preset['defaults']
        self.set_entries(data)

    def importParams(self):
        caption = 'Open File'
        accepted = 'JSON (*.json)'
        fileName = QFileDialog.getOpenFileName(self, caption=caption, directory='', filter=accepted)
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

    def exportParams(self, filename='ecgdata.json'):
        a, b, evt, omega = self.get_entries()
        helper.export_json(filename, a, b, evt, omega)

    def importECG(self, filename):
        csvdata = helper.import_csv(filename)
        for row in csvdata:
            print(row)

    def parseParams(self, a, b, evt, omega):
        arr_a = nparr([float(i) for i in a])
        arr_b = nparr([float(i) for i in b])
        arr_evt = nparr([ne.evaluate(helper.pirepl(i), self.exprdict) for i in evt])
        f_omega = ne.evaluate(helper.pirepl(omega), self.exprdict)

        return (arr_a, arr_b, arr_evt, f_omega)

    def buildParams(self):
        try:
            arr_a, arr_b, arr_evt, f_omega = self.parseParams(*self.get_entries())
        except:
            self.show_build_err()
            return
        self.buildECG(arr_a, arr_b, arr_evt, f_omega)

    def buildECG(self, a, b, evt, omega):
        sol = helper.solve_ecg(a, b, evt, omega)
        self.ecgplot.clear()
        self.ecgplot.plot(sol.t, sol.y[2], 'b-')
        self.ecgplot.figure.canvas.draw()