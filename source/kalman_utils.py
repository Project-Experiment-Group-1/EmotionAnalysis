import cv2
import numpy as np


class LandmarkKalmanFilter:
    """
    针对单个特征点 (x, y) 的卡尔曼滤波器
    """

    def __init__(self, Q=0.001, R=10.0):
        # 4个状态变量: [x, y, dx, dy] (位置和速度)
        # 2个观测变量: [x, y] (Dlib检测到的位置)
        self.kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵 (F)
        # 假设匀速运动模型: x_new = x_old + dx, y_new = y_old + dy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # 观测矩阵 (H)
        # 我们只能观测到位置 x, y，无法直接观测速度
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # 过程噪声协方差 (Q) - 代表我们对模型预测的信任度
        # 值越小，系统越相信模型（即相信人脸运动是平滑的），抗抖动越强
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * Q

        # 测量噪声协方差 (R) - 代表我们对传感器(Dlib)的信任度
        # 值越大，系统越认为观测值有噪点，平滑效果越强，但可能有延迟
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * R

        # 误差协方差 (P) - 初始化
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        # 标记是否是第一帧（第一帧需要直接初始化位置，不能从0开始平滑）
        self.first_run = True

    def update(self, measurement):
        """
        更新滤波器
        measurement: (x, y) 坐标
        """
        measured_point = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])

        if self.first_run:
            # 如果是第一帧，直接将状态设置为当前观测值，避免从(0,0)飞过来的过程
            self.kf.statePost = np.array([
                [measured_point[0][0]],
                [measured_point[1][0]],
                [0],
                [0]
            ], dtype=np.float32)
            self.first_run = False
            return measurement

        # 1. 预测 (Predict)
        self.kf.predict()

        # 2. 更新 (Correct)
        estimated = self.kf.correct(measured_point)

        return (estimated[0][0], estimated[1][0])


class LandmarksStabilizer:
    """
    管理所有68个特征点的滤波器
    """

    def __init__(self, num_landmarks=68, Q=0.001, R=10.0):
        self.num_landmarks = num_landmarks
        # 为每一个特征点创建一个独立的卡尔曼滤波器
        self.filters = [LandmarkKalmanFilter(Q, R) for _ in range(num_landmarks)]

    def update(self, landmarks_np):
        """
        输入: landmarks_np - numpy array shape (68, 2)
        输出: smoothed_landmarks - numpy array shape (68, 2)
        """
        if landmarks_np is None:
            return None

        smoothed_landmarks = np.zeros_like(landmarks_np, dtype=np.float32)

        for i in range(self.num_landmarks):
            # 获取第 i 个点的原始坐标
            raw_pt = landmarks_np[i]
            # 更新对应的滤波器
            smooth_pt = self.filters[i].update(raw_pt)
            # 保存平滑后的坐标
            smoothed_landmarks[i] = smooth_pt

        return smoothed_landmarks


def dlib_to_numpy(shape):
    """
    辅助函数：将dlib的full_object_detection对象转换为numpy数组
    """
    coords = np.zeros((68, 2), dtype=np.float32)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords