#!/usr/bin/env python3
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QLineEdit, QVBoxLayout)


class KalmanFilterForm(QDialog):
    """Form for Kalman Filtering
    Checkboxes to allow initial chosen values or values in form
    Checkboxes for plot or step-by-step output
    Other potential values, such as noise
    """

    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Extended Kalman Filter")
        self.setMinimumSize(550, 50)

        varianceBox = QGroupBox("Kalman Filter options")
        self.entry = QLineEdit("1")
        self.entry.setFixedWidth(30)
        self.entry.setMaxLength(5)
        t_validator = QRegExpValidator(QRegExp(r"\d"))
        self.entry.setValidator(t_validator)

        varianceLayout = QFormLayout(varianceBox)
        varianceLayout.addRow("Initial uncertainty:", self.entry)
        varianceBox.setLayout(varianceLayout)

        buttons = QDialogButtonBox(Qt.Horizontal, self)
        buttons.addButton(QDialogButtonBox.Ok)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        formlayout = QVBoxLayout(self)
        formlayout.addWidget(varianceBox)
        formlayout.addWidget(buttons)

    def get_variance(self):
        return int(self.entry.text())

    @staticmethod
    def get_ekf_options(parent=None):
        dialog = KalmanFilterForm(parent)
        result = dialog.exec_()
        variance = dialog.get_variance()
        return (variance, result == QDialog.Accepted)
