import cv2
import mediapipe as mp
import threading
import numpy as np
import keras
from multiprocessing import Process, Value, Array
import os
import win32gui, win32console
import tensorflow as tf
import queue

tf.config.experimental.set_visible_devices([], 'GPU')
MODEL = keras.models.load_model(
    './model_save/my_model_63.h5'
)
QUEUE_SIZE = 30

# 1. 메인 프로세스
def process_cam(array_for_static, value_for_static):
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    q = []
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # LR_idx = results.multi_handedness[i].classification[0].label
            if results.multi_hand_landmarks:
                for i in range(len(results.multi_handedness)):
                    # if results.multi_handedness[i].classification[0].label == 'Right':
                    hand_landmarks = results.multi_hand_landmarks[i]
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    out = np.zeros((21,3))
                    for i, res in enumerate(hand_landmarks.landmark):
                        if i == 0:
                            origin = out[i]
                        out[i] = [res.x, res.y, res.z] - origin
                    # print(out.flatten())

                    array_for_static[:] = out.flatten().tolist()
            now = value_for_static.value
            q.append(now)
            if len(q) > QUEUE_SIZE:
                q.pop(0)
            check = [0 for _ in range(32)]
            for i in q:
                check[i] += 1
            if max(check) > int(QUEUE_SIZE/2) and check.index(max(check)) != 0:
                print(check.index(max(check)))
                q = []

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

# 2. 손 모양 판별 프로세스
def process_static_gesture(array_for_static, value_for_static):
    while True:
        input_ = np.copy(array_for_static[:])
        # print(input_)
        input_ = input_[np.newaxis]
        # time.sleep(0.033)
        try:
            prediction = MODEL.predict(input_)
            if np.max(prediction[0]) > 0.5:
                value_for_static.value = np.argmax(prediction[0])
            else:
                value_for_static.value = 0
        except:
            pass

# initialize processes

if __name__ == "__main__":

    win32gui.ShowWindow(win32console.GetConsoleWindow(), 0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("Copyright 2021. INBODY inc. all rights reserved")
    print("Contact : shi@inbody.com, HAIL SONG")

    #physical_devices = tf.config.list_physical_devices('GPU')
    #print(physical_devices)
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # width = 1024 # 너비
    # height= 600 # 높이
    shared_array = Array('d', [0. for _ in range(63)])
    static_num = Value('i', 0)

    # GUI과 주고받을 정보 : Mode(쌍방향) [int value], 대기 [int value], 캡쳐 [int value], 이미지 [array]

    process1 = Process(target=process_cam, args=(shared_array, static_num))
    process2 = Process(target=process_static_gesture, args=(shared_array, static_num))

    process1.start()
    process2.start()

    while process1.is_alive():
        pass
    process2.terminate()

    process1.join()
    process2.join()
