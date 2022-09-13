import sys

from PyQt5 import QtCore, QtWidgets
import keyboard
from PyQt5.QtWidgets import QApplication


class KeyGrabber(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.hook = keyboard.on_press(self.keyboardEventReceived)

    def keyboardEventReceived(self, event):
        if event.event_type == 'down':
            if event.name == 'f3':
                print('F3 pressed')
            elif event.name == 'f4':
                print('F4 pressed')

def main():
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    _app = KeyGrabber()
    _app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
