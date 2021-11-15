import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QGridLayout, QTabWidget, QWidget
from inference_gui import Inference
from train_gui import Train

class TabWidget(QWidget):

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('steel scrap prediction')
        self.setWindowIcon(QIcon('smc_logo.png'))
        
        self.Inference = Inference()
        self.Train = Train()

        Tab = QTabWidget()
        Tab.addTab(self.Inference, 'Inference')
        Tab.addTab(self.Train, 'Train')

        layout = QGridLayout()
        layout.addWidget(Tab)
        self.setLayout(layout)

stylesheet = """
    MainWindow {
        background-image: url("C:/Users/Peerakarn/Desktop/smc_model/sys_steel_scrap_project/smc_logo.png"); 
        background-repeat: repeat; 
        background-position: center;
    }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    window = TabWidget()
    window.show()
    sys.exit(app.exec_())


