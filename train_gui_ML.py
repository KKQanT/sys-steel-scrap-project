import sys
from PyQt5.QtWidgets import QComboBox, QFormLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from configparser import ConfigParser
import pandas as pd
import numpy as np
import matplotlib
import time
import shutil
import os

matplotlib.use('QT5Agg')
matplotlib.rc('font', **{'size':8})

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import subprocess

from sklearn.metrics import mean_absolute_percentage_error


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Train