import cv2
import sys
import numpy as np
from PyQt5 import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.title = 'Temporary'
        self.geometry = (1280, 480, 670, 380)
        self.setCentralWidget(TemplateWindow(self))


class TemplateWindow(QWidget):
    def __init__(self, parent: MainWindow):
        super(TemplateWindow, self).__init__(parent)

        self.side_frame = QFrame(self)
        self.side_frame.setStyleSheet('QFrame{background-color:rgb(195, 130, 127);}')

        self.side_menu = QGridLayout(self.side_frame)
        self.side_menu.setAlignment(Qt.AlignTop | Qt.AlignBottom)

        self.side_info = QGridLayout(self.side_frame)
        self.side_info.setAlignment(Qt.AlignBottom)

        # SIDE WIDGETS
        self.side_menu.addWidget(QPushButton('Play'))
        self.side_menu.addWidget(QPushButton('Play'))
        self.side_info.addWidget(QPushButton('Cancel'))


def main():
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    _app = MainWindow()
    _app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
