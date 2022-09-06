import cv2
import mediapipe as mp
import math
import numpy as np
import re

import socket       # 发送数据

def cal_angle_3D(a, b, c):      # 计算关节三维角度
    x = cal_vector_3D(b, a)     # 是b->a还是a->b仍待商榷
    y = cal_vector_3D(b, c)

    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))

    dot = x.dot(y)
    cos_ = dot / (l_x * l_y)
    radian = np.arccos(cos_)
    angle = radian * 180 / np.pi

    return angle

def cal_vector_3D(x, y):        #计算两点间三维向量
    ans = []
    for i, j in zip(x, y):
        ans.append(j-i)

    return np.array(ans)

def joint_angles(hl):       # 计算手指每个关节的弯曲角度（三维）
    angle_dict = {}

    angle_dict['thumb_top'] = cal_angle_3D(hl[2], hl[3], hl[4])
    angle_dict['thumb_bottom'] = cal_angle_3D(hl[1], hl[2], hl[3])

    angle_dict['index_top'] = cal_angle_3D(hl[6], hl[7], hl[8])
    angle_dict['index_middle'] = cal_angle_3D(hl[5], hl[6], hl[7])
    angle_dict['index_bottom'] = cal_angle_3D(hl[0], hl[5], hl[6])

    angle_dict['middle_top'] = cal_angle_3D(hl[10], hl[11], hl[12])
    angle_dict['middle_middle'] = cal_angle_3D(hl[9], hl[10], hl[11])
    angle_dict['middle_bottom'] = cal_angle_3D(hl[0], hl[9], hl[10])

    angle_dict['ring_top'] = cal_angle_3D(hl[14], hl[15], hl[16])
    angle_dict['ring_middle'] = cal_angle_3D(hl[13], hl[14], hl[15])
    angle_dict['ring_bottom'] = cal_angle_3D(hl[0], hl[13], hl[14])

    angle_dict['pinky_top'] = cal_angle_3D(hl[18], hl[19], hl[20])
    angle_dict['pinky_middle'] = cal_angle_3D(hl[17], hl[18], hl[19])
    angle_dict['pinky_bottom'] = cal_angle_3D(hl[0], hl[17], hl[18])

    return angle_dict


def shoot(angle_dict):      # 食指与中指第二个关节中有一个弯曲
    if angle_dict['middle_middle'] + angle_dict['index_middle'] <= 315:
        return "1"
    else:
        return "0"

def thumb_up(angle_dict):   # 大拇指第一、二个关节同时弯曲
    if angle_dict['thumb_top'] > 145 and angle_dict['thumb_bottom'] > 160:
        print(angle_dict['define'] + '\tUP')
    else:
        print(angle_dict['define'] + '\tDOWN')

def gun(angle_dict):        # 无名指与小拇指第二个关节同时弯曲,且食指中指第二个关节同时伸直
    if angle_dict['ring_middle'] < 120 and angle_dict['pinky_middle'] < 120 \
            and angle_dict['index_middle'] > 130 and angle_dict['middle_middle'] > 130:
        print(angle_dict['define'] + '\tTRUE')
    else:
        print(angle_dict['define'] + '\tFALSE')

def finger_bend(angle_dict,threshold = 450):    # 检测五指是否弯曲
    state_dict = {}

    if angle_dict['thumb_top'] > 145 and angle_dict['thumb_bottom'] > 150:
        state_dict['thumb'] = True
    else:
        state_dict['thumb'] = False

    if angle_dict['index_top'] + angle_dict['index_middle'] + angle_dict['index_bottom'] > threshold:
        state_dict['index'] = True
    else:
        state_dict['index'] = False

    if angle_dict['middle_top'] + angle_dict['middle_middle'] + angle_dict['middle_bottom'] > threshold:
        state_dict['middle'] = True
    else:
        state_dict['middle'] = False

    if angle_dict['ring_top'] + angle_dict['ring_middle'] + angle_dict['ring_bottom'] > threshold:
        state_dict['ring'] = True
    else:
        state_dict['ring'] = False

    if angle_dict['pinky_top'] + angle_dict['pinky_middle'] + angle_dict['pinky_bottom'] > threshold:
        state_dict['pinky'] = True
    else:
        state_dict['pinky'] = False

    return state_dict

def five(angle_dict):       # 五指全部伸直
    state_dict = finger_bend(angle_dict,threshold=480)
    if False not in state_dict.values():
        print('Five')
    else:
        print('--')


def send_socket(trigger):
    # 1.创建socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 2. 链接服务器
    server_addr = ("127.0.0.1", 444)
    tcp_socket.connect(server_addr)

    # 3. 发送数据
    tcp_socket.send(trigger.encode("UTF-8"))

    # 4. 关闭套接字
    tcp_socket.close()


def get_landmarks():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            max_num_hands=1,
            min_tracking_confidence=0.5,) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)      #代入模型，获取结果

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks, multi_hands in zip(results.multi_hand_landmarks, results.multi_handedness):

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmark_style(),
                        drawing_styles.get_default_hand_connection_style())

                    landmarks_list = []     # 获取点坐标21*3的队列
                    for i in range(21):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        z = hand_landmarks.landmark[i].z

                        landmarks_list.append(np.array([x,y,z]))

                    angle_dict = joint_angles(landmarks_list)       # 计算关节弯曲角度的列表
                    angle_dict['define'] = str(multi_hands.classification).split('\n')[2].split('\"')[1]    # 获取左右手判别结果
                    # print(str(angle_dict['thumb_top']) + '\t' + str(angle_dict['thumb_bottom']))
                    trigger = shoot(angle_dict)     # 判别弹脑门的动作
            else:
                trigger = "-1"

            cv2.imshow('MediaPipe Hands', image)
            send_socket(trigger)        # 发送状态数据
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

if __name__ == '__main__':
    get_landmarks()