#!/usr/bin/env python3
from numpy import array as nparr

from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLineEdit, QVBoxLayout)


class KalmanFilterForm(QDialog):
    """Form for Kalman Filtering
    Checkboxes to allow initial chosen values or values in form
    Checkboxes for plot or step-by-step output
    Other potential values, such as noise
    """
    def __init__(self, parent=None):
        super().__init__()

        self.setWindowTitle("Parameter Estimation")
        self.setMinimumSize(550, 50)

        def make_entry(val, vdtr):
            entry = QLineEdit(str(val))
            entry.setFixedWidth(50)
            entry.setMaxLength(4)
            entry.setValidator(vdtr)
            return entry

        entries_a = QHBoxLayout()
        entries_b = QHBoxLayout()
        entries_evt = QHBoxLayout()
        num_validator = QRegExpValidator(QRegExp(r"\d*"))
        for i in range(0, 5):
            ai = make_entry(1, num_validator)
            bi = make_entry(1, num_validator)
            ei = make_entry(1, num_validator)

            entries_a.addWidget(ai)
            entries_b.addWidget(bi)
            entries_evt.addWidget(ei)

        entry_omega = make_entry(1, num_validator)

        varianceFrame = QGroupBox("Kalman Filter uncertainty")
        self.varianceForm = QFormLayout()
        self.varianceForm.addRow("a", entries_a)
        self.varianceForm.addRow("b", entries_b)
        self.varianceForm.addRow("theta", entries_evt)
        self.varianceForm.addRow("omega", entry_omega)
        varianceFrame.setLayout(self.varianceForm)

        buttons = QDialogButtonBox(Qt.Horizontal, self)
        buttons.addButton(QDialogButtonBox.Ok)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        frame = QVBoxLayout(self)
        frame.addWidget(varianceFrame)
        frame.addWidget(buttons)

    def get_variance(self):
        entries_a = self.varianceForm.itemAt(0, 1)
        entries_b = self.varianceForm.itemAt(1, 1)
        entries_evt = self.varianceForm.itemAt(2, 1)

        a_ = [entries_a.itemAt(i).widget().text().strip() for i in range(5)]
        b_ = [entries_b.itemAt(i).widget().text().strip() for i in range(5)]
        evt_ = [entries_evt.itemAt(i).widget().text().strip() for i in range(5)]
        omega_ = self.varianceForm.itemAt(3, 1).widget().text().strip()
        a = [str(1) if not ai else int(ai) for ai in a_]
        b = [str(1) if not bi else int(bi) for bi in b_]
        evt = [str(1) if not ei else int(ei) for ei in evt_]
        omega = str(1) if not omega_ else int(omega_)
        return nparr([1, 1, 1, *a, *b, *evt, omega])

    @staticmethod
    def get_ekf_options(parent=None):
        dialog = KalmanFilterForm(parent)
        result = dialog.exec_()
        variance = dialog.get_variance()
        return (variance, result == QDialog.Accepted)
