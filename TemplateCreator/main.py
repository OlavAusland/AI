import sys
from os.path import expanduser

import numpy
from PyQt5 import QtCore, Qt
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import requests
import urllib.request
import copy as cp
from scipy.spatial.distance import euclidean
import math


def dist(a: tuple, b: tuple) -> float:
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.title = 'Template Creator'
        self.geometry = (1280, 480, 670, 480)
        # --------------------------------------=
        self.init_header()

    def init_header(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.geometry[0], self.geometry[1], self.geometry[2], self.geometry[3])
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')
        edit_menu = main_menu.addMenu('Edit')
        view_menu = main_menu.addMenu('Windows')
        search_menu = main_menu.addMenu('Search')
        tools_menu = main_menu.addMenu('Tools')
        help_menu = main_menu.addMenu('Help')

        record_btn = QAction('Record Video', self)
        record_btn.setShortcut('Ctrl+R')
        record_btn.triggered.connect(self.set_record)
        view_menu.addAction(record_btn)

        template_btn = QAction('Template', self)
        template_btn.setShortcut('Ctrl+T')
        template_btn.triggered.connect(self.set_template)
        view_menu.addAction(template_btn)

        bounding_box_btn = QAction('Bounding Box', self)
        bounding_box_btn.setShortcut('Ctrl+B')
        bounding_box_btn.triggered.connect(self.set_bounding_box)
        view_menu.addAction(bounding_box_btn)

        home_btn = QAction('Home', self)
        home_btn.setShortcut('Ctrl+H')
        home_btn.triggered.connect(self.set_self)
        view_menu.addAction(home_btn)

        exit_btn = QAction(QIcon('exit24.png'), 'Exit', self)
        exit_btn.setShortcut('Ctrl+Q')
        exit_btn.setStatusTip('Exit application')
        exit_btn.triggered.connect(self.close)
        file_menu.addAction(exit_btn)

        self.show()

    def set_self(self):
        self.setCentralWidget(None)

    def set_record(self):
        self.setCentralWidget(RecordWindow(self))
        self.setWindowTitle('Template Creator - Record')

    def set_template(self):
        self.setCentralWidget(TemplateWindow(self))
        self.setWindowTitle('Template Creator - Template')

    def set_bounding_box(self):
        self.setCentralWidget(BoundingBoxWindow(self))
        self.setWindowTitle('Template Creator - Bounding Box')


class TemplateWindow(QWidget):
    def __init__(self, parent: MainWindow):
        super(TemplateWindow, self).__init__()
        self.container = QHBoxLayout(self)
        self.container.setContentsMargins(0, 0, 0, 0)
        self.container.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)

        self.side_frame = QFrame(self)
        self.side_frame.setStyleSheet('QFrame{background-color: rgb(200, 200, 200);}')
        self.side_menu = QGridLayout(self.side_frame)
        self.side_menu.setAlignment(Qt.AlignTop)

        self.main_frame = QFrame(self)
        self.main_menu = QVBoxLayout(self.main_frame)
        self.main_menu.setContentsMargins(0, 0, 0, 0)
        self.main_menu.setSpacing(0)

        # Widgets

        # row 0

        self.category_text = QLabel('Category:')
        self.category = QLineEdit("")

        # row 1

        self.input_text = QLabel('Input File:')
        self.file = QPushButton('undefined')
        self.file.clicked.connect(self.select_file)

        # - row 2

        self.folder_txt: str = 'C:/'
        self.output_text = QLabel('Output Folder:')
        self.out_folder = QPushButton('C:\\')
        self.out_folder.clicked.connect(self.select_folder)

        # row 3

        self.write_btn = QPushButton('Write All')
        self.write_btn.clicked.connect(self.write)

        self.write_frame_btn = QPushButton('Write Frame')
        self.next_frame_btn = QPushButton('Next')
        self.prev_frame_btn = QPushButton('Prev')

        # attach side menu
        self.side_menu.addWidget(self.category_text, 0, 0)
        self.side_menu.addWidget(self.category, 0, 1)
        self.side_menu.addWidget(self.input_text, 1, 0)
        self.side_menu.addWidget(self.file, 1, 1)
        self.side_menu.addWidget(self.output_text, 2, 0)
        self.side_menu.addWidget(self.out_folder, 2, 1)
        self.side_menu.addWidget(self.write_btn, 3, 0, 1, 2)
        self.side_menu.addWidget(self.write_frame_btn, 4, 0, 1, 0)
        self.side_menu.addWidget(self.prev_frame_btn, 5, 0)
        self.side_menu.addWidget(self.next_frame_btn, 5, 1)

        # main menu
        self.video = None
        self.preview = QLabel()
        self.progress = QProgressBar()
        self.progress.setValue(58)

        self.main_menu.addWidget(self.preview)
        self.main_menu.addWidget(self.progress)


        #final
        self.splitter.addWidget(self.side_frame)
        self.splitter.addWidget(self.main_frame)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([150, 150])

        self.container.addWidget(self.splitter)

    def select_file(self):
        file_dir = QFileDialog.getOpenFileName(self, 'Select File', 'C:\\', "Image files (*.mp4)")
        self.file.setText(file_dir[0].split('/')[-1])
        self.file.setToolTip(file_dir[0])
        self.video = cv2.VideoCapture(file_dir[0])
        try:
            _, frame = self.video.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.preview.setPixmap(QPixmap.fromImage(
                QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).scaled(640, 480,
                                                                                                Qt.KeepAspectRatio)))
        except Exception as error:
            print(error)

    def select_folder(self):
        out_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.folder_txt = out_dir
        self.out_folder.setText(str(out_dir.split('/')[-1]))

    def write(self):
        self.extractor = VideoExtractor(self)
        self.extractor.start()


class BoundingBoxWindow(QWidget):
    def __init__(self, parent):
        super(BoundingBoxWindow, self).__init__(parent)
        self.splitter = QSplitter(Qt.Horizontal)
        self.container = QHBoxLayout(self)
        self.container.setContentsMargins(0, 0, 0, 0)
        self.container.setSpacing(0)

        # side frame
        self.side_frame = QFrame(self)
        self.side_frame.setStyleSheet('QFrame{background-color:rgb(200, 200, 200);}')
        self.side_menu = QGridLayout(self.side_frame)
        self.side_menu.setAlignment(Qt.AlignTop)

        # main frame
        self.main_frame = QFrame(self)

        # main menu
        self.preview = QLabel(self.main_frame)
        self.preview.mousePressEvent = self.display_mouse

        # side menu
        self.file_text = QLabel('Video File:')
        self.file = QPushButton('undefined')
        self.file.clicked.connect(self.select_file)

        self.output_text = QLabel('Output File:')
        self.output_dir = QPushButton('C:/')
        self.output_dir.clicked.connect(self.select_dir)

        self.write_btn = QPushButton('write')
        self.prev_btn = QPushButton('previous')
        self.next_btn = QPushButton('next')

        self.side_menu.addWidget(self.file_text, 0, 0)
        self.side_menu.addWidget(self.file, 0, 1)
        self.side_menu.addWidget(self.output_text, 1, 0)
        self.side_menu.addWidget(self.output_dir, 1, 1)
        self.side_menu.addWidget(self.write_btn, 2, 0, 1, 0)
        self.side_menu.addWidget(self.prev_btn, 3, 0)
        self.side_menu.addWidget(self.next_btn, 3, 1)

        self.splitter.addWidget(self.side_frame)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([150, 150])

        self.container.addWidget(self.splitter)
        self.show()
        # feed
        self.feed = BoundingBox(self)

    def __del__(self):
        self.feed.stop()

    def start(self):
        self.feed.start()
        self.feed.ImageUpdate.connect(self.update_image)

    def update_image(self, image):
        self.preview.setPixmap(QPixmap.fromImage(image))

    def select_file(self):
        file_dir = QFileDialog.getOpenFileName(self, 'Select File', 'C:\\', "Image files (*.png)")
        self.file.setText(file_dir[0].split('/')[-1])
        self.file.setToolTip(file_dir[0])
        self.feed.image = cv2.imread(file_dir[0])
        self.start()
        try:
            image = cv2.imread(file_dir[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.preview.setPixmap(QPixmap.fromImage(
                QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).scaled(640, 480,
                                                                                                Qt.KeepAspectRatio)))
        except Exception as error:
            print(error)

    def select_dir(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.output_dir.setText(input_dir.split('/')[-1])
        self.output_dir.setToolTip(input_dir)

    def display_mouse(self, event):
        if self.feed is None:
            return
        if event.button() == QtCore.Qt.LeftButton:
            self.feed.p1 = (event.x(), event.y())
        else:
            self.feed.p2 = (event.x(), event.y())
        self.feed.update = True


class BoundingBox(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, parent: BoundingBoxWindow):
        super().__init__(parent)
        self.is_active = False
        self.image_file = parent.file
        self.p1: tuple = (0, 0)
        self.p2: tuple = (0, 0)
        self.image = None
        self.update = False

    def run(self):
        self.is_active = True
        image = None
        while self.is_active:
            if not self.update: continue
            image = cv2.copyTo(self.image, None, image)
            image = cv2.rectangle(image, self.p1, self.p2, color=[0, 0, 255], thickness=2)
            for point in [self.p1, self.p2]:
                image = cv2.circle(image, point, radius=3, color=[0, 255, 0], thickness=3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qt_image = QImage(image.data, image.shape[1], image.shape[0],
                              QImage.Format_RGB888)
            Pic = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)
            self.update = False

    def stop(self):
        self.is_active = False


class RecordWindow(QWidget):
    def __init__(self, parent):
        super(RecordWindow, self).__init__(parent)
        self.layout = QVBoxLayout()
        self.hlayout = QHBoxLayout()

        self.filename = QLineEdit("output.mp4")
        self.filename.deselect()
        self.hlayout.addWidget(self.filename, stretch=0)

        self.is_recording = False

        self.feed_label = QLabel()
        self.layout.addWidget(self.feed_label, alignment=Qt.AlignHCenter)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_record)
        self.record_btn.setDisabled(True)
        self.hlayout.addWidget(self.record_btn, stretch=1)

        self.choose_directory_btn = QPushButton('C:\\')
        self.choose_directory_btn.clicked.connect(self.select_dir)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)
        self.hlayout.addWidget(self.start_btn, stretch=1)

        self.layout.addLayout(self.hlayout)
        self.layout.addWidget(self.choose_directory_btn)

        self.feed = LiveFeed(self)

        self.setLayout(self.layout)

    def __del__(self):
        self.cancel()

    def get_mouse_pos(self, event):
        position: tuple = (event.pos().x(), event.pos().y())
        self.feed.position = position

        if cv2.waitKey(1) & 0xFF == ord('w'):
            self.feed.box_size = (self.feed.box_size[0] + 10, self.feed.box_size[1])
        elif cv2.waitKey(1) & 0xFF == ord('h'):
            self.feed.box_size = (self.feed.box_size[0], self.feed.box_size[1] + 10)

    def start(self):
        self.record_btn.setDisabled(False)
        self.start_btn.setText('Stop')
        self.start_btn.clicked.connect(self.cancel)
        self.feed.start()
        self.feed.ImageUpdate.connect(self.update_image)

    def select_dir(self):
        input_dir = QFileDialog.getExistingDirectory(None, 'Select a folder:', expanduser("~"))
        self.choose_directory_btn.setText(str(input_dir))
        del self.feed
        self.feed = LiveFeed(self)

    def toggle_record(self):
        self.is_recording = not self.is_recording

        if self.is_recording:
            self.record_btn.setText('Stop Recording')
        else:
            self.record_btn.setText('Start Recording')

    def update_image(self, image):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def cancel(self):
        self.start_btn.setText('Start')
        self.start_btn.clicked.connect(self.start)
        self.record_btn.setDisabled(True)
        self.feed.stop()


class VideoExtractor(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, parent: TemplateWindow):
        super().__init__(parent)
        self.cap: cv2.VideoCapture = parent.video
        self.category: str = parent.category.text()
        self.path: str = parent.folder_txt
        self.is_active: bool = False
        self.idx: int = 0

    def run(self):
        self.is_active = True
        print(f'{self.path}/{self.category}_{self.idx}.png')
        while self.cap.isOpened() and self.is_active:
            ret, frame = self.cap.read()
            if ret:
                image = cv2.flip(frame, 1)
                print("writing")
                cv2.imwrite(f'{self.path}/{self.category}_{self.idx}.png', image)
            else:
                self.is_active = False
                print("still inside")
            self.idx += 1
        self.cap.release()

    def stop(self):
        self.is_active = False


class LiveFeed(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, parent):
        super().__init__(parent)
        try:
            self.cap = cv2.VideoCapture(-1)
            self.path = parent.choose_directory_btn.text()
            self.filename = parent.filename.text()
            self.is_active = False
            self.font = cv2.FONT_HERSHEY_SIMPLEX

            # bounding box
            self.position: tuple = (20, 20)
            self.box_size: tuple = (60, 60)

        except Exception as error:
            print(error)

    def __del__(self):
        self.cap.release()
        print("CALLED DESTRUCTOR")

    def run(self):
        self.is_active = True

        cap = cv2.VideoCapture(0)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(f'{self.path}/{self.filename}', fourcc, 20.0, (int(w), int(h)))
        if not cap.isOpened(): return
        while self.is_active:
            print(self.position)
            ret, frame = cap.read()
            if ret:
                image = cv2.flip(frame, 1)

                if self.parent().is_recording:
                    print("recording...")
                    out.write(image)
                    image = cv2.putText(image, "RECORDING", color=[0, 0, 255], fontScale=1, thickness=2,
                                        fontFace=self.font, org=(0, 25))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, value=[150, 150, 150])

                qt_image = QImage(image.data, image.shape[1], image.shape[0],
                                  QImage.Format_RGB888)
                Pic = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
        empty = np.full((int(h), int(w), 3), 255, dtype=np.uint8)
        empty = cv2.putText(empty, "NO VIDEO!", color=[0, 0, 0], fontScale=1, fontFace=self.font, thickness=3,
                            org=(0, 25))
        empty = cv2.copyMakeBorder(empty, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, value=[150, 150, 150])
        self.ImageUpdate.emit(QImage(empty.data, empty.shape[1], empty.shape[0],
                                     QImage.Format_RGB888).scaled(640, 480, Qt.KeepAspectRatio))
        cap.release()
        out.release()

    def stop(self):
        self.is_active = False


def main():
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    _app = MainWindow()
    _app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
