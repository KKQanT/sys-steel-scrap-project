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

class Train3Months(QWidget):

    def __init__(self):
        super().__init__()

        self.defaults_params_dicts = {
            'taiwan_small_bigru_avgadj2':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"84",
                'param_n_units':"[4,4]",
                'param_middle_dense_dim':"None",
                'param_dropout':"0",
                'param_epochs':"300",

                'param_head_size':"-------",
                'param_num_heads':"-------",
                'param_ff_dim':"-------",
                'param_num_transformer_blocks':"-------",
                'param_mlp_units':"-------",
                "param_mlp_dropout":"-------"
            },
            'taiwan_gru_baseline_avg':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"2",
                'param_middle_dense_dim':'-------',
                "param_dropout":"",
                'param_epochs':'100',

                'param_head_size':"-------",
                'param_num_heads':"-------",
                'param_ff_dim':"-------",
                'param_num_transformer_blocks':"-------",
                'param_mlp_units':"-------",
                "param_mlp_dropout":"-------"
            },
            'domestic_baseline_gru_avg':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"2",
                'param_middle_dense_dim':'-------',
                "param_dropout":"-------",
                'param_epochs':'50',

                'param_head_size':"-------",
                'param_num_heads':"-------",
                'param_ff_dim':"-------",
                'param_num_transformer_blocks':"-------",
                'param_mlp_units':"-------",
                "param_mlp_dropout":"-------"
            },
            'domestic_bigru_avg':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"[8,8]",
                'param_middle_dense_dim':'None',
                "param_dropout":"0",
                'param_epochs':'300',

                'param_head_size':"-------",
                'param_num_heads':"-------",
                'param_ff_dim':"-------",
                'param_num_transformer_blocks':"-------",
                'param_mlp_units':"-------",
                "param_mlp_dropout":"-------"
            },
            'domestic_transformerv1_avgsel':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"-------",
                'param_middle_dense_dim':'-------',
                "param_dropout":"0.2",
                'param_epochs':'500',

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
            'taiwan_small_bigru_avgadj2',
            'taiwan_gru_baseline_avg',
            'domestic_baseline_gru_avg',
            'domestic_bigru_avg',
            'domestic_transformerv1_avgsel',
            ])
        self.select_model.currentTextChanged.connect(self.onSelectModelChanged)
        self.select_model.currentTextChanged.connect(self.updateGraph)
        self.select_model.currentTextChanged.connect(self.updateMape)
        select_model_layout = QVBoxLayout()
        select_model_layout.addWidget(self.select_model)


        self.param_split_pct = QLineEdit("20")
        self.param_epochs = QLineEdit('300')
        self.param_seed = QLineEdit("0")
        self.param_window = QLineEdit("84")
        self.param_dropout = QLineEdit('0')
        ########## small gru #####################
        self.param_n_units = QLineEdit("[4, 4]")
        self.param_middle_dense_dim = QLineEdit("None")
        ########## transformer ###################
        self.param_head_size = QLineEdit('')
        self.param_num_heads = QLineEdit('')
        self.param_ff_dim = QLineEdit('')
        self.param_num_transformer_blocks = QLineEdit('')
        self.param_mlp_units = QLineEdit('')
        self.param_mlp_dropout = QLineEdit('')

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

        try:
            self.df_val = pd.read_csv(f'output/{self.select_model.currentText()}_val.csv')
            self.df_test = pd.read_csv(f'output/{self.select_model.currentText()}_test.csv')
            self.df_val['target_date'] = pd.to_datetime(self.df_val['target_date'])
            self.df_test['target_date'] = pd.to_datetime(self.df_test['target_date'])
            self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
            self.canvas.axes.plot(self.df_val['target_date'], self.df_val['target'], label='actual', color='#16A085')
            self.canvas.axes.plot(self.df_val['target_date'], self.df_val['predict'], label='predict', color='#7D3C98')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['target'], color='#16A085')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['predict'], color='#7D3C98')
            self.canvas.axes.legend()
            self.canvas.axes.axvline(self.df_val['target_date'].max(), linestyle='dashed', color='red', alpha=0.5)

            val_mape = np.round(mean_absolute_percentage_error(self.df_val['target'], self.df_val['predict'])*100, decimals=1)
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
        self.param_seed.setText(new_params_dict['param_seed'])
        self.param_window.setText(new_params_dict['param_window'])
        self.param_dropout.setText(new_params_dict['param_dropout'])
        self.param_epochs.setText(new_params_dict['param_epochs'])

        self.param_n_units.setText(new_params_dict['param_n_units'])
        self.param_middle_dense_dim.setText(new_params_dict['param_middle_dense_dim'])

        self.param_head_size.setText(new_params_dict['param_head_size'])
        self.param_num_heads.setText(new_params_dict['param_num_heads'])
        self.param_ff_dim.setText(new_params_dict['param_ff_dim'])
        self.param_num_transformer_blocks.setText(new_params_dict['param_num_transformer_blocks'])
        self.param_mlp_units.setText(new_params_dict['param_mlp_units'])
        self.param_mlp_dropout.setText(new_params_dict['param_mlp_dropout'])

    def sendConfig(self):
        parser = ConfigParser()
        parser.read('src/deep_learning/model_config.ini')
        current_model = self.select_model.currentText().upper()
        parser.set(current_model, 'SEED', self.param_seed.text())
        parser.set(current_model, 'WINDOW', self.param_window.text())
        parser.set(current_model, 'DROPOUT', self.param_dropout.text())
        parser.set(current_model, 'EPOCHS', self.param_epochs.text())

        parser.set(current_model, 'N_UNITS', self.param_n_units.text())
        parser.set(current_model, 'MIDDLE_DENSE_DIM', self.param_middle_dense_dim.text())

        parser.set(current_model, 'HEAD_SIZE', self.param_head_size.text())
        parser.set(current_model, 'NUM_HEADS', self.param_num_heads.text())
        parser.set(current_model, 'FF_DIM', self.param_ff_dim.text())
        parser.set(current_model, 'NUM_TRANSFORMER_HEADS', self.param_num_transformer_blocks.text())
        parser.set(current_model, 'MLP_UNITS', self.param_mlp_units.text())
        parser.set(current_model, 'MLP_DROPOUT', self.param_mlp_dropout.text())

        with open("src/deep_learning/model_config.ini", 'w') as conf:
            parser.write(conf)
        
    def trainModel(self):

        self.sendConfig()

        time.sleep(2)

        subprocess.call('preprocessing.bat', shell=True)
        process = subprocess.Popen(
                [f'{self.select_model.currentText()}_train.bat'],
                cwd='src/deep_learning/',
                shell=True
            )
        process.communicate()
        time.sleep(2)
        self.updateGraph()
        self.updateMape()

    def updateGraph(self):

        try:
        
            self.df_val = pd.read_csv(f'output/{self.select_model.currentText()}_val.csv')
            self.df_test = pd.read_csv(f'output/{self.select_model.currentText()}_test.csv')
            self.df_val['target_date'] = pd.to_datetime(self.df_val['target_date'])
            self.df_test['target_date'] = pd.to_datetime(self.df_test['target_date'])
            
            self.canvas.axes.cla()
            self.canvas.axes.plot(self.df_val['target_date'], self.df_val['target'], label='actual', color='#16A085')
            self.canvas.axes.plot(self.df_val['target_date'], self.df_val['predict'], label='predict', color='#7D3C98')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['target'], color='#16A085')
            self.canvas.axes.plot(self.df_test['target_date'], self.df_test['predict'], color='#7D3C98')
            self.canvas.axes.legend()
            self.canvas.axes.axvline(self.df_val['target_date'].max(), linestyle='dashed', color='red', alpha=0.5)

            self.canvas.draw()

        except FileNotFoundError:
            self.canvas.axes.cla()
            self.canvas.draw()

    def updateMape(self):
        try:
            val_mape = np.round(mean_absolute_percentage_error(self.df_val['target'], self.df_val['predict'])*100, decimals=1)
            test_mape = np.round(mean_absolute_percentage_error(self.df_test['target'], self.df_test['predict'])*100, decimals=1)
            self.val_mape_text.setText(f"validatation set MAPE : {val_mape}")
            self.test_mape_text.setText(f"test set MAPE : {test_mape}")
        except AttributeError:
            self.val_mape_text.setText(f"validatation set MAPE : ")
            self.test_mape_text.setText(f"test set MAPE : ")

    def sendInferenceCConfig(self):
        parser = ConfigParser()
        parser.read('src/deep_learning/model_config.ini')
        current_model = self.select_model.currentText().upper()
        WINDOW = parser[current_model]['window']
        infer_parser = ConfigParser()
        infer_parser.read('src/deep_learning/infer_model_config.ini')
        infer_parser.set(current_model, 'window', WINDOW)
        with open("src/deep_learning/infer_model_config.ini", 'w') as conf:
            infer_parser.write(conf)

    def migrateModel(self):
        current_model = self.select_model.currentText()
        experiment = 'model/deep_learning/experiment/'
        executing = 'model/deep_learning/executing/'
        
        model_file = f'{current_model}.h5'
        val_date_file = f'{current_model}_val_date.pkl'
        scaler_X = f'{current_model}_scaler_X.pkl'
        scaler_y = f'{current_model}_scaler_y.pkl'

        for file in [model_file, val_date_file, scaler_X, scaler_y]:
            shutil.copyfile(os.path.join(experiment, file), os.path.join(executing, file))
