import sys
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QPushButton, QMainWindow, QGridLayout
from os.path import exists
import pandas as pd
import datetime as dt

import subprocess

class Inference(QWidget):

    def __init__(self):
        super().__init__()

        self._title = 'steel scrap inference'
        self.setWindowTitle(self._title)
        
        self.update_stooq_chkbox = QCheckBox("update steel scrap LME")
        self.update_stooq_chkbox.setChecked(True)

        self.update_yahoo_chkbox = QCheckBox("update yahoo data")
        self.update_yahoo_chkbox.setChecked(True)

        self.inference_ml = QCheckBox("inference - ML")
        self.inference_ml.setChecked(True)

        self.inference_dl_3month = QCheckBox("inference - DL (at the end of 3rd month)")
        self.inference_dl_3month.setChecked(True)

        self.inference_dl_1week = QCheckBox("inference - DL (domestic - in next week)")
        self.inference_dl_1week.setChecked(True)

        self.inference_dl_seq2seq = QCheckBox("inference - DL (domestic - long predict (1month and 3month)")
        self.inference_dl_seq2seq.setChecked(True)

        self.merge_output = QCheckBox('merge output for tableau')
        self.merge_output.setChecked(True)

        self.selectAllButton = QPushButton('Select All')
        self.selectAllButton.clicked.connect(self.onClickSelectAllButton)
        
        self.clearButton = QPushButton('Clear')
        self.clearButton.clicked.connect(self.onClickClearButton)

        self.okButton = QPushButton('OK')
        self.okButton.clicked.connect(self.onClickOkButton)

        chckbox_layout = QVBoxLayout()
        chckbox_layout.addWidget(self.update_stooq_chkbox)
        chckbox_layout.addWidget(self.update_yahoo_chkbox)
        chckbox_layout.addWidget(self.inference_ml)
        chckbox_layout.addWidget(self.inference_dl_3month)
        chckbox_layout.addWidget(self.inference_dl_1week)
        chckbox_layout.addWidget(self.inference_dl_seq2seq)
        chckbox_layout.addWidget(self.merge_output)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.selectAllButton)
        button_layout.addWidget(self.clearButton)
        button_layout.addWidget(self.okButton)

        self.status = QLabel("status")
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status)

        layout = QGridLayout()
        layout.addLayout(chckbox_layout, 0, 0)
        layout.addLayout(status_layout, 0, 1)
        layout.addLayout(button_layout, 1, 0)

        self.setLayout(layout)

    def onClickSelectAllButton(self):
        self.update_stooq_chkbox.setChecked(True)
        self.update_yahoo_chkbox.setChecked(True)
        self.inference_ml.setChecked(True)
        self.inference_dl_3month.setChecked(True)
        self.inference_dl_1week.setChecked(True)
        self.inference_dl_seq2seq.setChecked(True)
        self.merge_output.setChecked(True)

    def onClickClearButton(self):
        self.update_stooq_chkbox.setChecked(False)
        self.update_yahoo_chkbox.setChecked(False)
        self.inference_ml.setChecked(False)
        self.inference_dl_3month.setChecked(False)
        self.inference_dl_1week.setChecked(False)
        self.inference_dl_seq2seq.setChecked(False)
        self.merge_output.setChecked(False)

    def onClickOkButton(self):
        if self.update_stooq_chkbox.isChecked():
            self.status.setText('updating steel lme . . .')
            process = subprocess.Popen(['update_steel_lme.bat'])
            process.communicate()
            self.status.setText('updating steel lme . . . done')

        if self.update_yahoo_chkbox.isChecked():
            self.status.setText('downloading yahoo stock price . . .')
            process = subprocess.Popen(['yahoo_downloading.bat'])
            process.communicate()

            for i in range(5):

                self.status.setText('checking yahoos file . . .')
                data_detail, yahoo_file_completed = check_files()
                if yahoo_file_completed == True:
                    break
                else:
                    self.status.setText(f'downloading yahoo stock price . . . {i}/5')
                    process = subprocess.Popen(['yahoo_downloading.bat'])
                    process.communicate()

            self.status.setText('downloading yahoo stock price . . . done')

        if self.inference_ml.isChecked() or self.inference_dl_3month.isChecked() or self.inference_dl_1week.isChecked() or self.inference_dl_seq2seq.isChecked():
            self.status.setText('preprocessing target data . . .')
            process = subprocess.Popen(['preprocessing.bat'])
            process.communicate()
            self.status.setText('preprocessing target data . . .done')

        if self.inference_ml.isChecked():
            self.status.setText('machine learning model is forecasting scrap price at the next 3months . . .')
            process = subprocess.Popen(['ml_inference.bat'])
            process.communicate()
            self.status.setText('machine learning forecasting scrap price at the next 3months . . . done')

        if self.inference_dl_3month.isChecked():
            self.status.setText('deep learning model is forecasting scrap price at the next 3months . . .')
            process = subprocess.Popen(['inference_dl_3month.bat'])
            process.communicate()
            self.status.setText('deep learning model is forecasting scrap price at the next 3months . . . done')

        if self.inference_dl_1week.isChecked():
            self.status.setText('deep learning model is forecasting domestic scrap price at the next week . . .')
            process = subprocess.Popen(['inference_dl_1week.bat'])
            process.communicate()
            self.status.setText('deep learning model is forecasting domestic scrap price at the next week . . . Done')

        if self.inference_dl_seq2seq.isChecked():
            self.status.setText('deep learning model is forecasting domestic scrap price at 1week to 4/12 weeks . . .')
            process = subprocess.Popen(['inference_dl_seq2seq.bat'])
            process.communicate()
            self.status.setText('deep learning model is forecasting domestic scrap price at 1week to 4/12 weeks . . . Done')

        if self.merge_output.isChecked():
            self.status.setText('preprocessing data for tableau . . .')
            process = subprocess.Popen(['merge_output.bat'])
            process.communicate()
            self.status.setText('preprocessing data for tableau . . . done')

def check_files():

    data_detail = {}

    try:
        df = pd.read_csv('data/stooq/SteelScrapLME.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        data_detail['Steel_scrap_lme'] = (dt.datetime.strftime(min_date, "%d/%m/%Y"), 
                                    dt.datetime.strftime(max_date, "%d/%m/%Y"))
    
    except FileNotFoundError:
        data_detail['Steel_scrap_lme'] = 'not found'

    yahoo_file_completed = True
    yahoos_stocs_to_check = [
        '^TWII',
        '000333.SZ',
        '000932.SZ',
        '600019.SS',
        '601899.SS',
        'AH.BK',
        'MT',
        'SCHN',
        'TSM',
        'X'
    ]

    for file_name in yahoos_stocs_to_check:
        try:
            df = pd.read_csv(f'data/yahoo/{file_name}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            data_detail[file_name] = (dt.datetime.strftime(min_date, "%d/%m/%Y"), 
                                    dt.datetime.strftime(max_date, "%d/%m/%Y"))
            if len(df) < 1000:
                yahoo_file_completed = False
        except FileNotFoundError:
            data_detail[file_name] = 'not found'
            yahoo_file_completed = False
    
    return data_detail, yahoo_file_completed
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Inference()
    window.show()
    sys.exit(app.exec_())

