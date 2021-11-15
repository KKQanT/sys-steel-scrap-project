import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget, QCheckBox, QPushButton, QMainWindow, QGridLayout
from PyQt5.QtGui import QIcon

import subprocess

#class App(QWidget):
#
#    def __init__(self):
#        super().__init__()
#        self.title = 'Test'
#        self.left = 500
#        self.top = 250
#        self.width = 640
#        self.height = 480
#        self.initUI()
#
#    def initUI(self):
#        self.setWindowTitle(self.title)
#        self.setGeometry(self.left, self.top, self.width, self.height)
#
#        #self.getInteger()
#        #self.getText()
#        #self.getDouble()
#        #self.getChoice()
#        self.runPreprocessing()
#        #self.show()
#
#    def runPreprocessing(self):
#        text, okPressed = QInputDialog.getText(self, 'type something', 'params', QLineEdit.Normal, "")
#        if okPressed:
#            subprocess.call('preprocessing.bat')
#
#if __name__ == "__main__":
#    app = QApplication(sys.argv)
#    ex = App()
#    sys.exit(app.exec_())

class RunInference(QMainWindow):

    def __init__(self):
        super().__init__()

        self._title = 'steel scrap inference'
        self.setWindowTitle(self._title)

        self._main = QWidget()
        self.setCentralWidget(self._main)
        
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

        self.selectAllButton = QPushButton('Select All')
        self.clearButton = QPushButton('Clear')
        self.okButton = QPushButton('OK')

        chckbox_layout = QVBoxLayout()
        chckbox_layout.addWidget(self.update_stooq_chkbox)
        chckbox_layout.addWidget(self.update_yahoo_chkbox)
        chckbox_layout.addWidget(self.inference_ml)
        chckbox_layout.addWidget(self.inference_dl_3month)
        chckbox_layout.addWidget(self.inference_dl_1week)
        chckbox_layout.addWidget(self.inference_dl_seq2seq)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.selectAllButton)
        button_layout.addWidget(self.clearButton)
        button_layout.addWidget(self.okButton)

        layout = QGridLayout(self._main)
        layout.addLayout(chckbox_layout, 0, 0)
        layout.addLayout(button_layout, 1, 0)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = RunInference()
    ex.show()
    sys.exit(app.exec_())
