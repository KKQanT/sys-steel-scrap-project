import sys
from PyQt5 import QtWidgets
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

class TrainML(QWidget):
    def __init__(self):
        super().__init__()

        self.defaults_params_dicts = {
            'domestic':{
                'param_split_pct':"20",
                'param_window':"24",
                'param_threshold':"0.7",
                "param_std":"0.1",
                "param_var":"0.97"
            },

            'taiwan':{
                'param_split_pct':"20",
                'param_window':"24",
                'param_threshold':"0.7",
                "param_std":"0.1",
                "param_var":"0.95"
            }
        }

        self.select_model = QComboBox()
        self.select_model.addItems([
            'domestic',
            'taiwan'
            ])

        self.select_model.currentTextChanged.connect(self.onSelectModelChanged)
        self.select_model.currentTextChanged.connect(self.updateGraph)
        self.select_model.currentTextChanged.connect(self.updateMape)
        select_model_layout = QVBoxLayout()
        select_model_layout.addWidget(self.select_model)


        self.param_split_pct = QLineEdit("20")
        self.param_window = QLineEdit("24")
        self.param_threshold = QLineEdit("0.7")
        self.param_std = QLineEdit("0.1")
        self.param_var = QLineEdit("0.97")

        param_layout = QFormLayout()
        param_layout.addRow("percentage of val+test", self.param_split_pct)
        param_layout.addRow('window', self.param_window)
        param_layout.addRow("threshold", self.param_threshold)
        param_layout.addRow("std", self.param_std)
        param_layout.addRow('var', self.param_var)

        try:
            self.df_test_all = pd.read_csv('output/domestic_ml_test_all.csv')
            self.df_test = pd.read_csv('output/domestic_ml_test.csv')
            self.df_test_all['target_date'] = pd.to_datetime(self.df_test_all['target_date'])
            self.df_test['target_date'] = pd.to_datetime(self.df_test['target_date'])
            self.canvas = MplCanvas(self, width=7, height=3, dpi=100)
            self.canvas.axes.plot(self.df_test_all['target_date'], self.df_test_all['target'], label='actual', color='#16A085')
            self.canvas.axes.plot(self.df_test_all['target_date'], self.df_test_all['predict'], label='predict', color='#7D3C98')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['target'], color='#16A085')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['predict'], color='#7D3C98')
            self.canvas.axes.legend()
            self.canvas.axes.axvline(self.df_test_all['target_date'].max(), linestyle='dashed', color='red', alpha=0.5)

            val_mape = np.round(mean_absolute_percentage_error(self.df_test_all['target'], self.df_test_all['predict'])*100, decimals=1)
            test_mape = np.round(mean_absolute_percentage_error(self.df_test['target'], self.df_test['predict'])*100, decimals=1)

            self.val_mape_text = QLabel(f"validatation set MAPE : {val_mape}")
            self.test_mape_text = QLabel(f"test set MAPE : {test_mape}")

        except FileNotFoundError:
            self.canvas = MplCanvas(self, width=5, height=3, dpi=100)

            self.val_mape_text = QLabel(f"validatation set MAPE : ")
            self.test_mape_text = QLabel(f"test set MAPE : ")
        
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(self.canvas)

        performance_layout = QVBoxLayout()
        performance_layout.addWidget(self.val_mape_text)
        performance_layout.addWidget(self.test_mape_text)

        train_button = QPushButton("train")
        train_button.clicked.connect(self.trainModel)
        train_layout = QVBoxLayout()
        train_layout.addWidget(train_button)

        migration_button = QPushButton("replace previous model")
        migration_button.clicked.connect(self.sendInferenceCConfig)
        migration_button.clicked.connect(self.migrateModel)
        migration_button_layout = QVBoxLayout()
        migration_button_layout.addWidget(migration_button)

        layout = QGridLayout()
        layout.addLayout(graph_layout, 0, 0)
        layout.addLayout(performance_layout, 1, 0)
        layout.addLayout(migration_button_layout, 2, 0)
        layout.addLayout(select_model_layout, 0, 1)
        layout.addLayout(param_layout, 1, 1)
        layout.addLayout(train_layout, 2, 1)
        self.setLayout(layout)

    def onSelectModelChanged(self, value):
        new_model = self.select_model.currentText()
        
        new_params_dict = self.defaults_params_dicts[new_model]

        self.param_split_pct.setText(new_params_dict['param_split_pct'])
        self.param_window.setText(new_params_dict['param_window'])
        self.param_threshold.setText(new_params_dict['param_threshold'])
        self.param_std.setText(new_params_dict['self.param_std'])
        self.param_var.setText(new_params_dict['self.param_var'])

    def sendConfig(self):
        parser = ConfigParser()
        parser.read('src/machine_learning/model_config.ini')
        current_model = self.select_model.currentText().upper()
        parser.set(current_model, 'SPLIT_PCT', self.param_split_pct.text())
        parser.set(current_model, 'WINDOW', self.param_window.text())
        parser.set(current_model, 'THRESHOLD', self.param_threshold.text())
        parser.set(current_model, 'STD', self.param_std.text())
        parser.set(current_model, 'VAR', self.param_var.text())

        with open("src/machine_learning/model_config.ini", 'w') as conf:
            parser.write(conf)

    def trainModel(self):

        self.sendConfig()

        time.sleep(2)

        subprocess.call('preprocessing.bat', shell=True)
        process = subprocess.Popen(
                [f'{self.select_model.currentText()}_train_and_inference.bat'],
                cwd='src/machine_learning/',
                shell=True
            )
        process.communicate()
        time.sleep(2)
        self.updateGraph()
        self.updateMape()

    def updateGraph(self):

        try:
        
            self.df_test_all = pd.read_csv(f'output/{self.select_model.currentText()}_ml_test_all.csv')
            self.df_test = pd.read_csv(f'output/{self.select_model.currentText()}_ml_test.csv')
            
            self.canvas.axes.cla()

            self.df_test_all['target_date'] = pd.to_datetime(self.df_test_all['target_date'])
            self.df_test['target_date'] = pd.to_datetime(self.df_test['target_date'])
            self.canvas = MplCanvas(self, width=7, height=3, dpi=100)
            self.canvas.axes.plot(self.df_test_all['target_date'], self.df_test_all['target'], label='actual', color='#16A085')
            self.canvas.axes.plot(self.df_test_all['target_date'], self.df_test_all['predict'], label='predict', color='#7D3C98')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['target'], color='#16A085')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['predict'], color='#7D3C98')
            self.canvas.axes.legend()
            self.canvas.axes.axvline(self.df_test_all['target_date'].max(), linestyle='dashed', color='red', alpha=0.5)


            self.canvas.draw()

        except FileNotFoundError:
            self.canvas.axes.cla()
            self.canvas.draw()

    def updateMape(self):
        try:
            val_mape = np.round(mean_absolute_percentage_error(self.df_test_all['target'], self.df_test_all['predict'])*100, decimals=1)
            test_mape = np.round(mean_absolute_percentage_error(self.df_test['target'], self.df_test['predict'])*100, decimals=1)
            self.val_mape_text.setText(f"validatation set MAPE : {val_mape}")
            self.test_mape_text.setText(f"test set MAPE : {test_mape}")
        except AttributeError:
            self.val_mape_text.setText(f"validatation set MAPE : ")
            self.test_mape_text.setText(f"test set MAPE : ")

    def sendInferenceCConfig(self):
        parser = ConfigParser()
        parser.read('src/machine_learning/model_config.ini')
        current_model = self.select_model.currentText().upper()
        WINDOW = parser[current_model]['window']
        infer_parser = ConfigParser()
        infer_parser.read('src/machine_learning/infer_model_config.ini')
        infer_parser.set(current_model, 'window', WINDOW)
        with open("src/machine_learning/infer_model_config.ini", 'w') as conf:
            infer_parser.write(conf)

    def migrateModel(self):
        current_model = self.select_model.currentText()
        experiment = 'model/machine_learning/experiment/'
        executing = 'model/machine_learning/executing/'
        
        model_file = f'{current_model}.h5'
        val_date_file = f'{current_model}_val_date.pkl'
        scaler_X = f'{current_model}_scaler_X.pkl'
        scaler_y = f'{current_model}_scaler_y.pkl'

        for file in [model_file, val_date_file, scaler_X, scaler_y]:
            shutil.copyfile(os.path.join(experiment, file), os.path.join(executing, file))