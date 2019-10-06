#!/usr/bin/env python3
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QGroupBox, QLabel,
                             QSlider, QVBoxLayout)


class SliderDialog(QDialog):
    def __init__(self, parent=None, tmin=1, tmax=1):
        super().__init__(parent)
        self.tmin = tmin
        self.tmax = tmax

        self.setWindowTitle('Select timeframe')
        self.setMinimumSize(550, 200)

        layout = QVBoxLayout(self)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(tmin)
        self.slider.setMaximum(tmax)
        self.slider.setValue(tmin)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        # self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.valueChanged.connect(self.valuechange)

        self.label = QLabel("t = {} (s)".format(self.tmin))

        buttons = QDialogButtonBox(Qt.Horizontal, self)
        buttons.addButton(QDialogButtonBox.Ok)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        sliderlayout = QVBoxLayout()
        sliderlayout.addWidget(self.label)
        sliderlayout.addWidget(self.slider)
        sliderlayout.addWidget(buttons)

        groupbox = QGroupBox('Select timeframe of sample:')
        groupbox.setLayout(sliderlayout)
        layout.addWidget(groupbox)

    def reset(self):
        self.label.setText(str(self.tmin))
        self.slider.setValue(self.tmin)

    def valuechange(self):
        tframe = self.slider.value()
        self.label.setText("t = {} (s)".format(tframe))

    def timeFrame(self):
        return self.slider.value()

    @staticmethod
    def getTimeFrame(parent=None, tmin=1, tmax=1):
        dialog = SliderDialog(parent, tmin, tmax)
        result = dialog.exec_()
        tframe = dialog.timeFrame()
        return (tframe, result == QDialog.Accepted)
