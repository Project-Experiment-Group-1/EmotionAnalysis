#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script gives a real-time demonstration of the facial expression analysis
model, and updates a simple set of graphs. It makes use of your camera and
analyses your face in real-time. DLIB landmarks are not very illumination
invariant, so this works better when there are no shadows on the face.
"""

import numpy as np
import cv2
import dlib
from emotions_dlib import EmotionsDlib

import time

GRAPH_LENGTH = 30  # frames
EXP_AVG = 0.5  # 70% the new value 30% the old value

cap = cv2.VideoCapture(0)  # camera object
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

emotion_estimator = EmotionsDlib(
    file_emotion_model='../models/model_emotion_pls=30_fullfeatures=False.joblib',
    file_frontalization_model='../models/model_frontalization.npy'
)



f = 0
points_av = None
points_arousal = None
polyg_arousal = None
points_valence = None
polyg_valence = None
points_intensity = None
polyg_intensity = None

ls_arousal = []
ls_valence = []
ls_intensity = []

ls_captures = []

disp_arousal = 0
disp_valence = 0
disp_intensity = 0

if __name__ == "__main__":

    while True:  # for 300 frames
        # print('frame', f)
        f += 1
        # TODO: find a better way of closing this window. At the moment it
        # keeps going!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Q pressed')
            cap.release()
            cv2.destroyAllWindows()
            f = 0
            break

        ret, frame = cap.read()  # capture a frame

        cv2.putText(
            img=frame,
            text="press q to quit",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        cv2.imshow("Camera Feed", frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(image)  # detect faces

        if len(faces) > 0:  # if there are faces detected

            # find larger detected face and select it
            face_size = 0
            idx_largest_face = 0
            if len(faces) > 1:
                for i, face in enumerate(faces):
                    current_size = ((face.bottom() - face.top()) *
                                    (face.right() - face.left()))
                    if face_size < current_size:
                        face_size = current_size
                        idx_largest_face = i

            face = faces[idx_largest_face]  # the face to be processed
            landmarks_object = predictor(image, face)  # detect landmarks
            dict_emotions = emotion_estimator.get_emotions(  # get emotions
                landmarks_object
            )

            # parse output
            landmarks = dict_emotions['landmarks']['raw']
            landmarks_frontal = dict_emotions['landmarks']['frontal']
            arousal = dict_emotions['emotions']['arousal']
            valence = dict_emotions['emotions']['valence']
            intensity = dict_emotions['emotions']['intensity']
            emotion_name = dict_emotions['emotions']['name']

            print(
                f"Valence: {valence:.2f} | "
                f"Arousal: {arousal:.2f} | "
                f"Intensity: {intensity:.2f} | "
                f"Emotion: {emotion_name}"
            )

            # explonential averaging
            disp_valence = EXP_AVG * valence + (1 - EXP_AVG) * disp_valence
            disp_arousal = EXP_AVG * arousal + (1 - EXP_AVG) * disp_arousal
            disp_intensity = EXP_AVG * intensity + (1 - EXP_AVG) * disp_intensity

            # append buffer lists
            ls_arousal.append(disp_arousal)
            ls_valence.append(disp_valence)
            ls_intensity.append(disp_intensity)
            if len(ls_arousal) > GRAPH_LENGTH:
                del ls_arousal[0]
                del ls_valence[0]
                del ls_intensity[0]
            ls_xs = [i for i in range(len(ls_arousal))]
            ls_0z = [0 for i in range(len(ls_arousal))]

            f += 1

    # When everything done, release resources
    cap.release()
    cv2.destroyAllWindows()