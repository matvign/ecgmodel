#!/usr/bin/env python3
import sys
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QApplication

from gui import ecgapp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    ecgprog = ecgapp.ECGModel()
    sys.exit(app.exec())