import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QGridLayout, QTabWidget, QWidget
from inference_gui import Inference
from train_gui_3month import Train3Months
from train_gui_1week import Train1Week
from train_gui_seq2seq import TrainSeq2Seq
from train_gui_ML import TrainML

class TabWidget(QWidget):

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('steel scrap prediction')
        self.setWindowIcon(QIcon('smc_logo.png'))
        
        self.Inference = Inference()
        self.Train3Months = Train3Months()
        self.Train1Week = Train1Week()
        self.TrainSeq2Seq = TrainSeq2Seq()
        self.TrainML = TrainML()

        Tab = QTabWidget()
        Tab.addTab(self.Inference, 'Inference')
        Tab.addTab(self.Train3Months, 'Train3Months')
        Tab.addTab(self.Train1Week, 'Train1Week')
        Tab.addTab(self.TrainSeq2Seq, 'TrainSeq2Seq')
        Tab.addTab(self.TrainML, 'TrainML')

        layout = QGridLayout()
        layout.addWidget(Tab)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TabWidget()
    window.show()
    sys.exit(app.exec_())


