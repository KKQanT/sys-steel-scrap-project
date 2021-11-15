import sys
from PyQt5.QtWidgets import QApplication, QGridLayout, QTabWidget, QWidget
from inference_gui import Inference
from train_gui import Train

class TabWidget(QWidget):

    def __init__(self):
        super().__init__()
                
        self.Inference = Inference()
        self.Train = Train()

        Tab = QTabWidget()
        Tab.addTab(self.Inference, 'Inference')
        Tab.addTab(self.Train, 'Train')

        layout = QGridLayout()
        layout.addWidget(Tab)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TabWidget()
    window.show()
    sys.exit(app.exec_())


