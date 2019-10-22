#!/usr/bin/env python3
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QGroupBox, QVBoxLayout,
    QRadioButton)


class KalmanFilterForm(QDialog):
    """Form for Kalman Filtering
    Checkboxes to allow initial chosen values or values in form
    Checkboxes for plot or step-by-step output
    Other potential values, such as noise
    """

    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Extended Kalman Filter")
        self.setMinimumSize(550, 500)

        self.paramOpt = 0
        paramBox = QGroupBox("Parameter options:")
        paramLayout = QVBoxLayout(paramBox)
        initialOption = QRadioButton("Use pre-determined initial values", self)
        initialOption.setChecked(True)
        dynamOption = QRadioButton("Use dynamic parameters from form", self)
        paramLayout.addWidget(initialOption)
        paramLayout.addWidget(dynamOption)
        paramBox.setLayout(paramLayout)

        initialOption.toggled.connect(lambda: self.paramMode(0))
        dynamOption.toggled.connect(lambda: self.paramMode(1))

        self.solveOpt = 0
        solveBox = QGroupBox("Step options:")
        solveLayout = QVBoxLayout(solveBox)
        plotOption = QRadioButton("Plot EKF result", self)
        plotOption.setChecked(True)
        stepOption = QRadioButton("Plot EKF step-by-step", self)
        solveLayout.addWidget(plotOption)
        solveLayout.addWidget(stepOption)
        solveBox.setLayout(solveLayout)

        plotOption.toggled.connect(lambda: self.solveMode(0))
        stepOption.toggled.connect(lambda: self.solveMode(1))

        noiseBox = QGroupBox("Noise options:")

        buttons = QDialogButtonBox(Qt.Horizontal, self)
        buttons.addButton(QDialogButtonBox.Ok)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        formlayout = QVBoxLayout(self)
        formlayout.addWidget(paramBox)
        formlayout.addWidget(solveBox)
        formlayout.addWidget(buttons)

    def paramMode(self, opt):
        self.paramOpt = opt

    def solveMode(self, opt):
        self.solveOpt = opt

    def get_options(self):
        return (self.paramOpt, self.solveOpt)

    @staticmethod
    def get_ekf_options(parent=None):
        dialog = KalmanFilterForm(parent)
        result = dialog.exec_()
        options = dialog.get_options()
        return (options, result == QDialog.Accepted)
