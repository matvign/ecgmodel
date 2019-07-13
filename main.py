#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication

from gui import ecgapp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ecgprog = ecgapp.ECGModel()
    sys.exit(app.exec())