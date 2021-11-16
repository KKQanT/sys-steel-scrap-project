import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from configparser import ConfigParser
import pandas as pd
import matplotlib

matplotlib.use('QT5Agg')
matplotlib.rc('font', **{'size':8})

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=10, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class Train(QWidget):

    def __init__(self):
        super().__init__()

        self.defaults_params_dicts = {
            'taiwan_small_bigru_avgadj2':{
                'param_split_pct':"20",
                'param_seed':"0",
                'param_window':"84",
                'param_n_units':"[4,4]",
                'param_middle_dense_dim':"None"
            },
            'taiwan_gru_baseline_avg':{
                'param_split_pct':"",
                'param_seed':"0",
                'param_window':"168",
                'param_n_units':"2",
                'param_middle_dense_dim':''
            }
        }
        self.select_model = QComboBox()
        self.select_model.addItems([
            'taiwan_small_bigru_avgadj2',
            'taiwan_gru_baseline_avg',
            ])
        self.select_model.currentTextChanged.connect(self.onSelectModelChanged)
        select_model_layout = QVBoxLayout()
        select_model_layout.addWidget(self.select_model)


        self.param_split_pct = QLineEdit("20")
        self.param_seed = QLineEdit("0")
        self.param_window = QLineEdit("84")
        self.param_n_units = QLineEdit("[4, 4]")
        self.param_middle_dense_dim = QLineEdit("None")
        param_layout = QFormLayout()
        param_layout.addRow("percentage of val+test", self.param_split_pct)
        param_layout.addRow("random seed", self.param_seed)
        param_layout.addRow("lookback window (day)", self.param_window)
        param_layout.addRow("gru units", self.param_n_units)
        param_layout.addRow("dimension of middle dense", self.param_middle_dense_dim)


        self.df_val = pd.read_csv(f'output/{self.select_model.currentText()}_val.csv')
        self.df_test = pd.read_csv(f'output/{self.select_model.currentText()}_test.csv')
        self.df_val['target_date'] = pd.to_datetime(self.df_val['target_date'])
        self.df_test['target_date'] = pd.to_datetime(self.df_test['target_date'])

        sc = MplCanvas(self, width=5, height=3, dpi=100)
        sc.axes.plot(self.df_val['target_date'], self.df_val['target'], label='actual', color='#16A085')
        sc.axes.plot(self.df_val['target_date'], self.df_val['predict'], label='predict', color='#7D3C98')
        sc.axes.plot(self.df_test['target_date'], self.df_test['target'], color='#16A085')
        sc.axes.plot(self.df_test['target_date'], self.df_test['predict'], color='#7D3C98')
        sc.axes.legend()
        sc.axes.axvline(self.df_val['target_date'].max(), linestyle='dashed', color='red', alpha=0.5)

        graph_layout = QVBoxLayout()
        graph_layout.addWidget(sc)

        train_button = QPushButton("train")
        train_button.clicked.connect(self.train_model)
        train_layout = QVBoxLayout()
        train_layout.addWidget(train_button)

        layout = QGridLayout()
        layout.addLayout(graph_layout, 0, 0)
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
        self.param_n_units.setText(new_params_dict['param_n_units'])
        self.param_middle_dense_dim.setText(new_params_dict['param_middle_dense_dim'])

    def train_model(self):
        config_object = ConfigParser()
        config_object['MODEL_PARAMS'] = {
            "SPLIT_PCT":self.param_split_pct.text(),
            "SEED": self.param_seed.text(),
            "WINDOW":self.param_window.text(),
            "N_UNITS":self.param_n_units.text(),
            "MIDDLE_DENSE_DIM":self.param_middle_dense_dim.text(),
        }

        with open("src/deep_learning/model_config.ini", 'w') as conf:
            config_object.write(conf)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Train()
    window.show()
    sys.exit(app.exec_())