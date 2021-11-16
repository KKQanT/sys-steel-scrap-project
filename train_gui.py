import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QComboBox, QFormLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from configparser import ConfigParser

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
        label = QLabel()
        pixmap  = QPixmap('output/output.png')
        label.resize(1000, 300)
        label.setPixmap(pixmap.scaled(label.size(), QtCore.Qt.KeepAspectRatio))

        vray = QVBoxLayout()
        vray.addWidget(label)

        train_button = QPushButton("train")
        train_button.clicked.connect(self.train_model)
        
        layout = QGridLayout()
        layout.addLayout(vray, 0, 0)
        layout.addLayout(select_model_layout, 0, 1)
        layout.addLayout(param_layout, 1, 1)

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
            "N_UNITS":self.param_n_uits.text(),
            "MIDDLE_DENSE_DIM":self.param_middle_dense_dim.text(),
        }

        with open("model_config.ini", 'w') as conf:
            config_object.write(conf)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Train()
    window.show()
    sys.exit(app.exec_())