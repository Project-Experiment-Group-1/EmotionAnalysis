#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vasileios Vonikakis (modified by @GGGT)
@email: bbonik@gmail.com   (Vonikakis)

This script gives a real-time demonstration of the facial expression analysis
model, and updates a simple set of graphs. It makes use of your camera and
analyses your face in real-time. DLIB landmarks are not very illumination
invariant, so this works better when there are no shadows on the face.
"""

import numpy as np
import cv2
import dlib
import matplotlib

# [FIX] 强制使用非交互式后端 'Agg'。
# 这必须在导入 pyplot 之前或初始化时完成，防止 Matplotlib 抢占 GUI 事件循环。
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emotions_dlib import EmotionsDlib, plot_landmarks
from kalman_utils import LandmarksStabilizer, dlib_to_numpy  # 导入我们新建的卡尔曼工具
import math


class FrameAnalyzer:
    """analyze face of every frame"""

    def __init__(self,
                 smoothing_factor=0.8,
                 deadzone_threshold=0.05,
                 calibration_frames=60,  # frames used for calibration
                 prefictor_path=f'../models/shape_predictor_68_face_landmarks.dat',
                 file_emotion_model=f'../models/model_emotion_fromMorphset_pls=31_fullfeatures=False.joblib',
                 file_frontalization_model=f'../models/model_frontalization.npy'):
        self.smoothing_factor = smoothing_factor
        self.deadzone_threshold = deadzone_threshold

        self.calibration_frames = calibration_frames
        self._calibration_buffer = []
        self._is_calibrating = False

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(prefictor_path)
        self.emotion_estimator = EmotionsDlib(file_emotion_model=file_emotion_model,
                                              file_frontalization_model=file_frontalization_model)

        # initialize Kalman filter
        self.stabilizer = LandmarksStabilizer(num_landmarks=68, Q=0.001, R=5.0)

        self.s_prev_vals = {'valence': 0, 'arousal': 0, 'intensity': 0, }
        self.baseline = {'valence': 0.0, 'arousal': 0.0, 'intensity': 0.0}

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(image)

        if len(faces) == 0:
            return None

        face = self._get_largest_face(faces)

        # get original landmarks and apply Kalman filter
        landmarks_object = self.predictor(image, face)
        landmarks_raw_np = dlib_to_numpy(landmarks_object)
        landmarks_smoothed_np = self.stabilizer.update(landmarks_raw_np)

        # estimate filtered landmarks
        dict_emotions = self.emotion_estimator.get_emotions(landmarks_smoothed_np)

        raw_results = {'landmarks': dict_emotions['landmarks']['raw'],
                       'landmarks_frontal': dict_emotions['landmarks']['frontal'],
                       'arousal': dict_emotions['emotions']['arousal'],
                       'valence': dict_emotions['emotions']['valence'],
                       'intensity': dict_emotions['emotions']['intensity'],
                       'name': dict_emotions['emotions']['name'],
                       'face_rect': face}

        s_results = self._apply_smoothing(raw_results)

        if self._is_calibrating:
            self._calibration_buffer.append(s_results)

            if len(self._calibration_buffer) > self.calibration_frames:
                self._finalize_calibration()

            s_results['is_corrected'] = False
            return s_results

        corrected_s_results = self._apply_correction(s_results)
        corrected_s_results['is_corrected'] = True

        return corrected_s_results

    def _get_largest_face(self, faces):
        face_size = 0
        idx_largest_face = 0

        if len(faces) > 1:
            for i, face in enumerate(faces):
                current_size = ((face.bottom() - face.top()) * (face.right() - face.left()))
                if face_size < current_size:
                    face_size = current_size
                    idx_largest_face = i

        return faces[idx_largest_face]

    def _apply_smoothing(self, current_results):
        alpha = self.smoothing_factor
        for key in ['valence', 'arousal', 'intensity']:
            s_current_val = alpha * current_results[key] + (1 - alpha) * self.s_prev_vals[key]
            self.s_prev_vals[key] = s_current_val
            current_results[key] = s_current_val

        return current_results

    def start_calibration(self):
        self._is_calibrating = True
        self._calibration_buffer = []

    def is_calibrating(self):
        return self._is_calibrating

    def get_calibration_progress(self):
        """progress of (0.0 - 1.0)"""
        if not self._is_calibrating:
            return 0.0
        return len(self._calibration_buffer) / self.calibration_frames

    def _finalize_calibration(self):
        """calculate and apply baseline"""
        count = len(self._calibration_buffer)
        if count == 0:
            return

        avg_v = sum(r['valence'] for r in self._calibration_buffer) / count
        avg_a = sum(r['arousal'] for r in self._calibration_buffer) / count
        avg_i = sum(r['intensity'] for r in self._calibration_buffer) / count

        self.baseline['valence'] = avg_v
        self.baseline['arousal'] = avg_a
        self.baseline['intensity'] = avg_i

        self.s_prev_vals = {'valence': 0, 'arousal': 0, 'intensity': 0, }
        self._calibration_buffer = []
        self._is_calibrating = False
        print(f"Calibration Complete. New Baseline -> V:{avg_v:.2f}, A:{avg_a:.2f}")

    def _apply_correction(self, s_results):
        """apply calibration and deadzone"""
        v_corrected = s_results['valence'] - self.baseline['valence']
        if v_corrected > 1.0:
            v_corrected = 1.0
        elif v_corrected < -1.0:
            v_corrected = -1.0

        if abs(v_corrected) < self.deadzone_threshold:
            v_corrected = 0.0

        a_corrected = s_results['arousal'] - self.baseline['arousal']
        if a_corrected > 1.0:
            a_corrected = 1.0
        elif a_corrected < -1.0:
            a_corrected = -1.0

        if abs(a_corrected) < self.deadzone_threshold:
            a_corrected = 0.0

        i_corrected = math.sqrt(v_corrected ** 2 + a_corrected ** 2)
        if i_corrected > 1:
            i_corrected = 1

        name = self.avi_to_text(a_corrected, v_corrected, i_corrected)

        s_results['valence'] = v_corrected
        s_results['arousal'] = a_corrected
        s_results['intensity'] = i_corrected
        s_results['name'] = name

        return s_results

    def avi_to_text(self, arousal, valence, intensity=None):
        '''
        Generates a text description for a pair of arousal-valence values
        based on Russell's Circulmplex Model of Affect.
        Russell, J. A. (1980). A circumplex model of affect. Journal of
        Personality and Social Psychology, 39(6), 1161–1178.
        '''

        expression_intensity = "?"
        expression_name = "?"

        if intensity is None: intensity = math.sqrt(arousal ** 2 + valence ** 2)

        ls_expr_intensity = [
            "Slightly", "Moderately", "Very", "Extremely"
        ]
        ls_expr_name = [
            "pleased", "happy", "delighted", "excited", "astonished",
            "aroused",  # first quarter

            "tensed", "alarmed", "afraid", "annoyed", "distressed",
            "frustrated", "miserable",  # second quarter

            "sad", "gloomy", "depressed", "bored", "droopy", "tired",
            "sleepy",  # third quarter

            "calm", "serene", "content", "satisfied"  # fourth quarter
        ]

        # analyzing intensity
        if intensity < 0.1:
            expression_name = "Neutral"
            expression_intensity = ""
        else:

            if intensity < 0.325:
                expression_intensity = ls_expr_intensity[0]
            elif intensity < 0.55:
                expression_intensity = ls_expr_intensity[1]
            elif intensity < 0.775:
                expression_intensity = ls_expr_intensity[2]
            else:
                expression_intensity = ls_expr_intensity[3]

            # analyzing epxression name

            # compute angle [0,360]
            if valence == 0:
                if arousal >= 0:
                    theta = 90
                else:
                    theta = 270
            else:
                theta = math.atan(arousal / valence)
                theta = theta * (180 / math.pi)

                if valence < 0:
                    theta = 180 + theta
                elif arousal < 0:
                    theta = 360 + theta

            # estimate expression name

            if theta < 16 or theta > 354:
                expression_name = ls_expr_name[0]
            elif theta < 34:
                expression_name = ls_expr_name[1]
            elif theta < 62.5:
                expression_name = ls_expr_name[2]
            elif theta < 78.5:
                expression_name = ls_expr_name[3]
            elif theta < 93:
                expression_name = ls_expr_name[4]
            elif theta < 104:
                expression_name = ls_expr_name[5]
            elif theta < 115:
                expression_name = ls_expr_name[6]
            elif theta < 126:
                expression_name = ls_expr_name[7]
            elif theta < 137:
                expression_name = ls_expr_name[8]
            elif theta < 148:
                expression_name = ls_expr_name[9]
            elif theta < 159:
                expression_name = ls_expr_name[10]
            elif theta < 170:
                expression_name = ls_expr_name[11]
            elif theta < 181:
                expression_name = ls_expr_name[12]
            elif theta < 192:
                expression_name = ls_expr_name[13]
            elif theta < 203:
                expression_name = ls_expr_name[14]
            elif theta < 215:
                expression_name = ls_expr_name[15]
            elif theta < 230:
                expression_name = ls_expr_name[16]
            elif theta < 245:
                expression_name = ls_expr_name[17]
            elif theta < 260:
                expression_name = ls_expr_name[18]
            elif theta < 280:
                expression_name = ls_expr_name[19]
            elif theta < 300:
                expression_name = ls_expr_name[20]
            elif theta < 320:
                expression_name = ls_expr_name[21]
            elif theta < 340:
                expression_name = ls_expr_name[22]
            elif theta < 354:
                expression_name = ls_expr_name[23]
            else:
                expression_name = "Unknown"
                expression_intensity = ""

        # TODO: return also variable output and not only string

        return expression_intensity + " " + expression_name


class Visualizer:
    def __init__(self, show_cam=True, show_graph=True, graph_length=30):
        self.show_cam = show_cam
        self.show_graph = show_graph
        self.graph_length = graph_length

        # [FIX] 移除 plt.ion()，不再使用 Matplotlib 的交互模式
        # plt.ion()
        plt.style.use('seaborn')

        self.fig = plt.figure(figsize=(15, 10))
        self.grid = plt.GridSpec(5, 6)
        # 移除 Matplotlib 标题中的按键提示，因为现在窗口分离了，逻辑略有不同
        plt.suptitle('Emotion Analysis Graphs')

        self._init_plots()

        # initialize value list for waveform
        self.ls_arousal = []
        self.ls_valence = []
        self.ls_intensity = []

        # objects for graph
        self.points_av = None
        self.points_series = {}
        self.polys_series = {}

        # set camera feed name
        self.window_name = "Camera Feed"

        # [FIX] 新增：用于存储图表图像的变量
        self.graph_image = None

        # [FIX] 移除 Matplotlib 的键盘事件监听，全部交由 OpenCV 处理
        # self.mpl_key_buffer = None
        # self.fig.canvas.mpl_connect('key_press_event', self._on_mpl_key_press)

    def _init_plots(self):
        """initialize subplots and axes"""

        # subplot1: original landmarks
        self.ax_orig = plt.subplot(self.grid[:2, :2])
        self.ax_orig.set_title('Original landmarks')

        # subplot2: frontalized landmarks
        self.ax_front = plt.subplot(self.grid[:2, 2:4])
        self.ax_front.set_title('Frontalized landmarks')

        # subplot3: AV space
        self.ax_av = plt.subplot(self.grid[:2, 4:])
        self.ax_av.set_title('Arousal Valence Space')
        self.ax_av.set_xlim((-1, 1))
        self.ax_av.set_ylim((-1, 1))
        self.ax_av.set_xlabel('Valence')
        self.ax_av.set_ylabel('Arousal')
        self.ax_av.axhline(linewidth=3, color='k')
        self.ax_av.axvline(linewidth=3, color='k')
        self.ax_av.grid(True)
        deadzone_rect = plt.Rectangle((-0.15, -0.15), 0.3, 0.3, fill=False, edgecolor='gray', linestyle='--')
        self.ax_av.add_patch(deadzone_rect)

        # subplots4, 5, 6: waveform
        self.axes_series = {}
        labels = ['Valence', 'Arousal', 'Intensity']
        rows = [2, 3, 4]

        self.colors = {'Valence': (0.8, 0.2, 0.2), 'Arousal': (0.2, 0.2, 0.8), 'Intensity': (0.2, 0.8, 0.2)}

        for idx, label in zip(rows, labels):
            # take evry col of row idx for each waveform
            ax = plt.subplot(self.grid[idx, :])
            ax.set_ylabel(label)

            if label == 'Intensity':
                ax.set_ylim((0, 1.01))
            else:
                ax.set_ylim((-1.01, 1.01))

            ax.set_xlim((0, self.graph_length - 1))
            ax.set_xticks([])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1]) if label != 'Intensity' else [0, 0.5, 1]
            self.axes_series[label] = ax

    def update_charts(self, result_data):
        """update charts by data from RealtimeEmotionAnalysis for every frame"""
        if result_data is None:
            return

        landmarks = result_data['landmarks']
        landmarks_frontal = result_data['landmarks_frontal']
        valence = result_data['valence']
        arousal = result_data['arousal']
        intensity = result_data['intensity']
        name = result_data['name']
        if not result_data.get('is_corrected', True):
            name += " (Raw)"

        self.ls_valence.append(valence)
        self.ls_arousal.append(arousal)
        self.ls_intensity.append(intensity)

        if len(self.ls_valence) > self.graph_length:
            self.ls_valence.pop(0)
            self.ls_arousal.pop(0)
            self.ls_intensity.pop(0)

        if not self.show_graph:
            return

        # create x-axis list and y-axis basis list(0 list)
        ls_xs = list(range(len(self.ls_valence)))
        ls_0z = [0] * len(self.ls_valence)

        # update AV space
        self.ax_orig.clear()
        plot_landmarks(landmarks, axis=self.ax_orig, title='Original landmarks')

        self.ax_front.clear()
        plot_landmarks(landmarks_frontal, axis=self.ax_front, title='Frontalized landmarks')

        # update points
        if self.points_av is not None:
            self.points_av.remove()
        self.points_av, = self.ax_av.plot(valence, arousal, color='r', marker='.', markersize=20)
        self.ax_av.set_title(f'AR={arousal:.2f} | VA={valence:.2f} | IN={intensity:.2f}\n{name}')

        # update waveform
        data_map = {'Valence': self.ls_valence, 'Arousal': self.ls_arousal, 'Intensity': self.ls_intensity}

        for label, data_list in data_map.items():
            ax = self.axes_series[label]
            color = self.colors[label]

            # clean lines
            if label in self.points_series and self.points_series[label] is not None:
                self.points_series[label].remove()
            # clean poly color
            if label in self.polys_series and self.polys_series[label] is not None:
                self.polys_series[label].remove()

            # draw new lines
            self.points_series[label], = ax.plot(data_list, linewidth=1, color=color)

            # fill new poly color
            self.polys_series[label] = ax.fill_between(ls_xs, data_list, ls_0z, interpolate=True, alpha=0.3,
                                                       color=color)

        # [FIX] 核心修复：不使用 plt.pause() 或 flush_events()
        # 而是将 Matplotlib 的画布渲染成内存中的一张图片，然后交给 OpenCV 显示
        self.fig.canvas.draw()

        # 从 canvas 获取图像数据 (RGB)
        try:
            # 这种方法兼容性最好
            img_w, img_h = self.fig.canvas.get_width_height()
            # 注意：tostring_rgb 较老，但为了兼容性通常保留。如果报错可尝试 buffer_rgba
            buf = self.fig.canvas.tostring_rgb()
            img_vis = np.frombuffer(buf, dtype=np.uint8)
            img_vis = img_vis.reshape((img_h, img_w, 3))

            # 转换为 BGR (OpenCV 格式)
            self.graph_image = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error converting plot to image: {e}")
            self.graph_image = None

    def show_camera_feed(self, frame, result_data, calibration_progress=None):
        if calibration_progress is not None and calibration_progress > 0:
            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            text = f"CALIBRATING... {int(calibration_progress * 100)}%"
            cv2.putText(frame, text, (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "KEEP A NEUTRAL FACE", (w // 2 - 180, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
        elif result_data is not None:
            rect = result_data['face_rect']
            name = result_data['name']

            # draw rectangle
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)

            # draw emotion name
            cv2.putText(frame, name, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'Q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'C' to Calibrate", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # 窗口1：摄像头画面
        if self.show_cam:
            cv2.imshow(self.window_name, frame)

        # [FIX] 窗口2：数据图表 (使用 OpenCV 显示，而不是 Matplotlib 窗口)
        if self.graph_image is not None:
            cv2.imshow("Analysis Graphs", self.graph_image)

        # 统一的按键检测 (OpenCV)
        cv_key = -1
        k = cv2.waitKey(1) & 0xFF
        if k != 255:  # 255: no key
            cv_key = k

        final_action = None
        if cv_key == ord('q'):
            final_action = 'quit'
        elif cv_key == ord('c'):
            final_action = 'calibrate'

        return final_action

    def close(self):
        plt.close('all')
        cv2.destroyAllWindows()


class RealtimeEmotionAnalysis:
    """output result on Console"""

    def print_results(self, result_data):
        if result_data is None:
            return

        valence = result_data['valence']
        arousal = result_data['arousal']
        intensity = result_data['intensity']
        name = result_data['name']

        print(
            f'Valence: {valence:.2f} | '
            f'Arousal: {arousal:.2f} | '
            f'Intensity: {intensity:.2f} | '
            f'Emotion: {name}'
        )


def run_demo(show_cam=True, show_graph=True):
    try:
        analyzer = FrameAnalyzer(deadzone_threshold=0.05, calibration_frames=60)
        # [MODIFIED] 将参数传递给 Visualizer
        visualizer = Visualizer(show_cam=show_cam, show_graph=show_graph)
        logger = RealtimeEmotionAnalysis()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting... Press 'C' to calibrate your face.")

    UPDATE_INTERVAL = 5  # update frequency (frame)
    frame_counter = 0
    last_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            # [NOTE] 保持了您之前的逻辑：校准时不跳帧
            if analyzer.is_calibrating() or (frame_counter % UPDATE_INTERVAL == 0):
                # face detect and emotion estimate
                last_result = analyzer.process_frame(frame)

                is_calibrating = analyzer.is_calibrating()

                # update chart when not calibrating
                if not is_calibrating:
                    visualizer.update_charts(last_result)

                # console
                logger.print_results(last_result)

            # update camera feed
            is_calibrating = analyzer.is_calibrating()
            calibration_progress = analyzer.get_calibration_progress() if is_calibrating else None

            action = visualizer.show_camera_feed(frame, last_result, calibration_progress)

            if action == 'quit':
                break
            elif action == 'calibrate':
                analyzer.start_calibration()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        visualizer.close()
        print("Resources released.")


if __name__ == '__main__':
    run_demo()