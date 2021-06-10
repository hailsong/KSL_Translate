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
'''
안녕하세요 이인권 교수님, 저는 올해 2월에 연세대학교 기계공학과를 졸업한 송하일이라고 합니다.

저는 기계공학과에서 학부생활을 하면서 컴퓨터 비전과 관련된 연구들에 크게 흥미를 느껴 학부 연구와 졸업을 위한 연구 논문, 졸업 작품 등에 컴퓨터 비전과 관련된 연구를 하거나 제품에 딥러닝을 적용하는 등의 시도를 했었습니다.
그러한 경험을 바탕으로 저는 작년 중순부터 인턴으로 일하던 회사에 졸업 직후 입사하여 컴퓨터비전과 관련된 프로젝트를 수행하고 있습니다.
하지만 저는 일을 하다가 제 배움이 아직 더 필요하다는 생각이 들었습니다.
개발을 위해 최신 논문들을 찾아볼 때 마다 멋진 논문을 써서 학계에 도움이 되면 정말 멋지겠다는 생각이 들던 중 김선주 교수님의 연구실 홈페이지를 찾게 되었고 이미지 센서를 통해 시스템이 세상을 인지하게 하는 연구들이 제가 하고싶던 일이라는 생각이 들었습니다.
할 수 있다면 좋은 저널들에 훌륭한 논문들을 쓰는 이교수님의 연구실에서 가르침을 받아 비전 분야의 멋진 연구를 꼭 해보고 싶다는 결론을 내렸습니다.

교수님께 연락을 드린 이유는 다름이 아니라 대학원 입학과 관련되어서 몇 가지 여쭤보고 싶어서인데 괜찮을까요?

저는 내년(2022년) 1학기에 컴퓨터과학과 석사로 대학원에 진학하길 희망하고 있고 혹시 가능하다면 교수님의 연구실에서 가르침을 받고 싶습니다.
혹시 연구원이 되기 위해 인턴 등 제가 거쳐야 할 과정들이 있을까요? 있다면 현재 티오가 있는지 알 수 있을까요?
비전 분야에서 수행하고있는 프로젝트들이나 성적증명 등 필요한 자료가 있으면 교수님께 보여드리고 싶습니다.

정말 많은 고민을 하다가 메일을 보냈습니다.
많이 부족하지만 메일에 간단히 작성한 CV를 첨부했습니다.
글로 제 이야기를 다 표현하기 힘들어 실례가 안된다면 한 번 찾아뵙거나 화상으로 조언을 듣고 싶습니다.
긴 글 읽어주셔서 정말 감사합니다!
건강 조심하시고 따뜻한 하루 보내시길 바랍니다.

연세대학교 졸업생 송하일 올림.
'''
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
