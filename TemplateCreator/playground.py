import sys

from PyQt5 import QtCore, QtWidgets
import keyboard
from PyQt5.QtWidgets import QApplication
import cv2


def main():
    cap = cv2.VideoCapture(-1)
    #cap = cv2.VideoCapture('C:/')

    while True:
        _, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
