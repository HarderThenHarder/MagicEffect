import sys
import cv2
from PyQt5 import QtWidgets, QtCore, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow

from mainForm import Ui_MainWindow


class MeituWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.camera = cv2.VideoCapture(0)
        self.is_open_camera = False
        self.effect_type = None

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.query_frame)
        self._timer.setInterval(30)
        self.classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def contour(self):
        self.effect_type = 'contour'

    def grab_image(self):
        if not self.is_open_camera:
            return

        self.grab = self.frame
        img_rows, img_cols, channels = self.grab.shape
        bytesPerLine = channels * img_cols
        QImg = QImage(self.grab.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.lbImageGrabbed.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbImageGrabbed.size()))

    def gray(self):
        self.effect_type = 'gray'

    def large_eyes(self):
        self.effect_type = 'large'

    def open_camera(self):
        self.is_open_camera = ~self.is_open_camera
        if self.is_open_camera:
            self.btnOpenCamera.setText("Close Camera")
            self._timer.start()
        else:
            self.btnOpenCamera.setText("Open Camera")
            self._timer.stop()

    def small_face(self):
        self.effect_type = 'face'

    def smooth(self):
        self.effect_type = 'smooth'

    def threshold(self):
        self.effect_type = 'threshold'

    def query_frame(self):
        ret, self.frame = self.camera.read()
        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols
        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.lbCamera.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbCamera.size()))

        if self.effect_type == 'gray':
            self.effect_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            img_rows, img_cols = self.effect_frame.shape
            bytesPerLine = img_cols
            QImg = QImage(self.effect_frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_Indexed8)
            self.lbBeautyImage.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbBeautyImage.size()))
        elif self.effect_type == 'smooth':
            self.effect_frame = cv2.GaussianBlur(self.frame, (5, 5), 50)
            img_rows, img_cols, channels = self.effect_frame.shape
            bytesPerLine = img_cols * channels
            QImg = QImage(self.effect_frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
            self.lbBeautyImage.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbBeautyImage.size()))
        elif self.effect_type == 'contour':
            self.effect_frame = cv2.Canny(self.frame, 30, 70)
            img_rows, img_cols = self.effect_frame.shape
            bytesPerLine = img_cols
            QImg = QImage(self.effect_frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_Indexed8)
            self.lbBeautyImage.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbBeautyImage.size()))
        elif self.effect_type == 'threshold':
            self.effect_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            _, self.effect_frame = cv2.threshold(self.effect_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_rows, img_cols = self.effect_frame.shape
            bytesPerLine = img_cols
            QImg = QImage(self.effect_frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_Indexed8)
            self.lbBeautyImage.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbBeautyImage.size()))
        elif self.effect_type == 'face':
            self.effect_frame = self.frame
            gray = cv2.cvtColor(self.effect_frame, cv2.COLOR_BGR2GRAY)
            face_rects = self.classfier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
            if len(face_rects) > 0:
                for face_rect in face_rects:
                    x, y, w, h = face_rect
                    cv2.rectangle(self.effect_frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 1)
            img_rows, img_cols, channels = self.effect_frame.shape
            bytesPerLine = img_cols * channels
            QImg = QImage(self.effect_frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
            self.lbBeautyImage.setPixmap(QPixmap.fromImage(QImg).scaled(self.lbBeautyImage.size()))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    meituApp = MeituWindow()
    meituApp.show()
    sys.exit(app.exec_())
