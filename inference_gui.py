import sys
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QPushButton, QMainWindow, QGridLayout

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

        layout = QGridLayout()
        layout.addLayout(chckbox_layout, 0, 0)
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
            subprocess.call('update_steel_lme.bat')
        if self.update_yahoo_chkbox.isChecked():
            subprocess.call('yahoo_downloading.bat')
        if self.inference_ml.isChecked() or self.inference_dl_3month.isChecked() or self.inference_dl_1week.isChecked() or self.inference_dl_seq2seq.isChecked():
            subprocess.call('preprocessing.bat')
        if self.inference_ml.isChecked():
            subprocess.call('ml_inference.bat')
        if self.inference_dl_3month.isChecked():
            subprocess.call('inference_dl_3month.bat')
        if self.inference_dl_1week.isChecked():
            subprocess.call('inference_dl_1week.bat')
        if self.inference_dl_seq2seq.isChecked():
            subprocess.call('inference_dl_seq2seq.bat')
        if self.merge_output.isChecked():
            subprocess.call('merge_output.bat')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Inference()
    window.show()
    sys.exit(app.exec_())

