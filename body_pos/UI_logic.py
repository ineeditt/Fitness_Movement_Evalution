import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow
from GUI import Ui_MainWindow
import Counter


class MainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.Model_init()
        self.SetConnect()

    def SetConnect(self):
        # 绑定槽函数
        self.shoulder_push.clicked.connect(lambda : self.Model_Process("Shoulder_Push"))
        self.flying_bird.clicked.connect(lambda : self.Model_Process("Flying_Bird"))
        self.squat.clicked.connect(lambda : self.Model_Process("Squat"))
        self.bend.clicked.connect(lambda : self.Model_Process("Bend"))
        self.back.clicked.connect(self.Back)
    def Model_init(self):
        # 检测模型
        file_model = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

        self.interpreter = tflite.Interpreter(model_path=file_model)
        self.interpreter.allocate_tensors()
        # 获取输入、输出的数据的信息
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # 获取PosNet 要求输入图像的高和宽
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        # 初始化帧率计算
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        # 初始化计数器
        self.counter = 0
        #初始化命令标志
        self.order = 0

    def Back(self):
        self.order = 0
        self.video = 0

    def Video_Change(self, type):
        if (type == "Shoulder_Push"):
            video = "video\Shoulder_Push.mp4"
        elif (type == "Flying_Bird"):
            video = "video\Flying_Bird.mp4"
        elif (type == "Bend"):
            video = "video/bend.mp4"
        elif (type == "Squat"):
            video = "video\squat.mp4"
         # 打开摄像头
        self.cap = cv2.VideoCapture(video)

    def Model_Process(self, type):
        self.Model_init()
        self.Video_Change(type)
        self.order = 1
        while(self.order):
            # 获取起始时间
            t1 = cv2.getTickCount()
            # 读取一帧图像
            success, img = self.cap.read()
            if not success:
                break
            # 获取图像帧的尺寸
            imH, imW, _ = np.shape(img)
            # 适当缩放
            img = cv2.resize(img, (int(330), int(600)))
            # 获取图像帧的尺寸
            imH, imW, _ = np.shape(img)
            # BGR 转RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 尺寸缩放适应PosNet 网络输入要求
            img_resized = cv2.resize(img_rgb, (self.width, self.height))
            # 维度扩张适应网络输入要求
            input_data = np.expand_dims(img_resized, axis=0)
            # 尺度缩放 变为 -1~+1
            input_data = (np.float32(input_data) - 128.0) / 128.0
            # 数据输入网络
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            # 进行关键点检测
            self.interpreter.invoke()
            # 获取hotmat
            hotmaps = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
            # 获取偏移量
            offsets = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
            # 获取hotmat的 宽 高 以及关键的数目
            h_output, w_output, n_KeyPoints = np.shape(hotmaps)
            # 存储关键点的具体位置
            keypoints = []
            # 关键点的置信度
            score = 0

            for i in range(n_KeyPoints):
                # 遍历每一张hotmap
                hotmap = hotmaps[:, :, i]

                # 获取最大值 和最大值的位置
                max_index = np.where(hotmap == np.max(hotmap))
                max_val = np.max(hotmap)

                # 获取y，x偏移量 前n_KeyPoints张图是y的偏移 后n_KeyPoints张图是x的偏移
                offset_y = offsets[max_index[0], max_index[1], i]
                offset_x = offsets[max_index[0], max_index[1], i + n_KeyPoints]

                # 计算在posnet输入图像中具体的坐标
                pos_y = max_index[0] / (h_output - 1) * self.height + offset_y
                pos_x = max_index[1] / (w_output - 1) * self.width + offset_x

                # 计算在源图像中的坐标
                pos_y = pos_y / (self.height - 1) * imH
                pos_x = pos_x / (self.width - 1) * imW

                # 取整获得keypoints的位置
                keypoints.append([int(round(pos_x[0])), int(round(pos_y[0]))])

                # 利用sigmoid函数计算置每一个点的置信度
                score = score + 1.0 / (1.0 + np.exp(-max_val))

            # 取平均得到最终的置信度
            score = score / n_KeyPoints

            if score > 0.5:
                # 标记关键点
                for point in keypoints:
                    cv2.circle(img, (point[0], point[1]), 5, (255, 255, 0), 5)

                # 画关节连接线
                # 左臂
                cv2.polylines(img, [np.array([keypoints[5], keypoints[7], keypoints[9]])], False, (0, 255, 0), 3)
                # # 右臂
                cv2.polylines(img, [np.array([keypoints[6], keypoints[8], keypoints[10]])], False, (0, 0, 255), 3)
                # # 左腿
                cv2.polylines(img, [np.array([keypoints[11], keypoints[13], keypoints[15]])], False, (0, 255, 0), 3)
                # # 右腿
                cv2.polylines(img, [np.array([keypoints[12], keypoints[14], keypoints[16]])], False, (0, 255, 255), 3)
                # 身体部分
                cv2.polylines(img, [np.array([keypoints[5], keypoints[6], keypoints[12], keypoints[11], keypoints[5]])],
                              False, (255, 255, 0), 3)

                # 更新计数器
                self.counter += Counter.train_counter(keypoints, type)

            # 显示计数
            cv2.putText(img, 'Counter: %d ' % self.counter, (imW - 350, imH - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示帧率
            cv2.putText(img, 'FPS: %.2f score:%.2f' % (self.frame_rate_calc, score), (imW - 350, imH - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            #将视频转到GUI界面输出
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_RGB888)
            self.label.setScaledContents(True)
            self.label.setPixmap(QtGui.QPixmap(frame))
            # # 显示结果
            # cv2.imshow('Pos', img)

            # 计算帧率
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / self.freq
            self.frame_rate_calc = 1 / time1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
