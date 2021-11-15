import sys
from PyQt5.QtWidgets import QApplication, QComboBox, QGridLayout, QVBoxLayout, QWidget

class Train(QWidget):

    def __init__(self):
        super().__init__()

        self.select_model = QComboBox()
        self.select_model.addItems([
            'taiwan gru baseline avg',
            'taiwan small bigru avgadj2' 
            ])
        self.select_model.activated.connect(self.current_text)

        select_model_layout = QVBoxLayout()
        select_model_layout.addWidget(self.select_model)

        layout = QGridLayout()
        layout.addLayout(select_model_layout, 0, 0)

        self.setLayout(layout)

    def current_text(self):
        ctext = self.select_model.currentText()
        print(ctext)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Train()
    window.show()
    sys.exit(app.exec_())