import sys
import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer


# 定义我们的主窗口类
class VisionSoftware(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("我的第一个工业视觉上位机")
        self.resize(800, 600)

        # 1. 搭建界面：一块幕布(用来显示画面) + 一个按钮
        self.video_label = QLabel("画面将显示在这里...")
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")

        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.setMinimumHeight(50)  # 让按钮变大一点

        # 把幕布和按钮像拼图一样上下排好
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_camera)

        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # 2. 核心魔法：信号与槽的 connect
        # 当按钮被点击(clicked)时，连接(connect)到开/关摄像头的函数
        self.btn_camera.clicked.connect(self.toggle_camera)

        # 3. 准备 OpenCV 相关的工具
        self.cap = None  # 用来存放摄像头对象
        self.timer = QTimer()  # 定时器：因为视频是一帧帧的图，我们需要一个定时器不断去刷图

        # 当定时器滴答响(timeout)时，连接(connect)到更新画面的函数
        self.timer.timeout.connect(self.update_frame)

    # 开关摄像头的逻辑
    def toggle_camera(self):
        if self.timer.isActive():  # 如果定时器在跑，说明摄像头开着
            self.timer.stop()
            self.cap.release()
            self.video_label.setText("摄像头已关闭")
            self.btn_camera.setText("打开摄像头")
        else:
            self.cap = cv2.VideoCapture(0)  # OpenCV 魔法：打开电脑默认摄像头(编号0)
            self.timer.start(30)  # 每隔 30 毫秒去截一张图 (1秒约30帧)
            self.btn_camera.setText("关闭摄像头")

    # 把 OpenCV 的画面贴到软件界面上的逻辑

    def update_frame(self):
        ret, frame = self.cap.read()  # 读取一帧画面
        if ret:
            # ================= 核心视觉处理区 开始 =================
            # 1. 灰度化和模糊
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)

            # 2. 二值化 (这里我把阈值设成了 100，你可以根据宿舍光线自己改)
            _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

            # 3. 找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 4. 过滤并画框
            count = 0
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # 只画面积大于 500 的轮廓
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # 画绿线
                    count += 1

            # 5. 在左上角实时显示数量
            cv2.putText(frame, f"Target Count: {count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # ================= 核心视觉处理区 结束 =================

            # 【最后一步】把处理完的 frame 转换颜色并显示到界面上
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img).scaled(self.video_label.size()))
# 启动软件的固定套路
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VisionSoftware()
    window.show()
    sys.exit(app.exec())