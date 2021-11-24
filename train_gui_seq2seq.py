import sys
from PyQt5.QtWidgets import QApplication, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
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

class TrainSeq2Seq(QWidget):

    def __init__(self):
        super().__init__()

        self.defaults_params_dicts = {
            'domestic_transformerv1_avgsel_week1_to_4':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"--------",
                'param_middle_dense_dim':"--------",
                'param_dropout':"0.2",
                'param_epochs':"500",

                'param_head_size':"256",
                'param_num_heads':"4",
                'param_ff_dim':"4",
                'param_num_transformer_blocks':"4",
                'param_mlp_units':"[32]",
                "param_mlp_dropout":"0.4"
            },

            'domestic_transformerv1_avgsel_week1_to_12':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"--------",
                'param_middle_dense_dim':"--------",
                'param_dropout':"0.2",
                'param_epochs':"500",

                'param_head_size':"256",
                'param_num_heads':"4",
                'param_ff_dim':"4",
                'param_num_transformer_blocks':"4",
                'param_mlp_units':"[32]",
                "param_mlp_dropout":"0.4"
            }
        }

        self.select_model = QComboBox()
        self.select_model.addItems([
            'domestic_transformerv1_avgsel_week1_to_4',
            'domestic_transformerv1_avgsel_week1_to_12'
            ])
        self.select_model.currentTextChanged.connect(self.onSelectModelChanged)
        self.select_model.currentTextChanged.connect(self.updateGraph)
        self.select_model.currentTextChanged.connect(self.updateMape)
        select_model_layout = QVBoxLayout()
        select_model_layout.addWidget(self.select_model)

        self.param_split_pct = QLineEdit("20")
        self.param_epochs = QLineEdit('500')
        self.param_seed = QLineEdit("0")
        self.param_window = QLineEdit("168")
        self.param_dropout = QLineEdit('0.2')
        ########## small gru #####################
        self.param_n_units = QLineEdit("-------")
        self.param_middle_dense_dim = QLineEdit("-------")
        ########## transformer ###################
        self.param_head_size = QLineEdit('256')
        self.param_num_heads = QLineEdit('4')
        self.param_ff_dim = QLineEdit('4')
        self.param_num_transformer_blocks = QLineEdit('4')
        self.param_mlp_units = QLineEdit('[32]')
        self.param_mlp_dropout = QLineEdit('0.4')

        param_layout = QFormLayout()
        param_layout.addRow("percentage of val+test", self.param_split_pct)
        param_layout.addRow('epochs', self.param_epochs)
        param_layout.addRow("random seed", self.param_seed)
        param_layout.addRow("lookback window (day)", self.param_window)
        param_layout.addRow('dropout', self.param_dropout)
        ########## small gru #####################
        param_layout.addRow("gru units", self.param_n_units)
        param_layout.addRow("dimension of middle dense", self.param_middle_dense_dim)
        ########## transfomer #####################
        param_layout.addRow('head size', self.param_head_size)
        param_layout.addRow('number of heads', self.param_num_heads)
        param_layout.addRow('fastforward dimension', self.param_ff_dim)
        param_layout.addRow('number of transformer blocks', self.param_num_transformer_blocks)
        param_layout.addRow('multi layer percepton units', self.param_mlp_units)
        param_layout.addRow('multi layer percepton dropout', self.param_mlp_dropout)
