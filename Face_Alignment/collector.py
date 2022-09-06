# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
import time

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader("<video0>")

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    flag = 0

    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)

            if len(boxes) == 0:
                continue

            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {args.opt}')

            left_box, right_box = cal_eyes_box(ver)

            img_draw = draw_box(left_box, img_draw)
            img_draw = draw_box(right_box, img_draw)

            left = frame[left_box[1][1]:left_box[0][1], left_box[2][0]:left_box[0][0]]
            left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)

            right = frame[right_box[1][1]:right_box[0][1], right_box[2][0]:right_box[0][0]]
            right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

            cv2.imshow('left', left)
            cv2.imshow('right',right)

            cv2.imshow('image', img_draw)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("w"):      #open
                print('Start collecting open.')
                flag = 1
            elif key & 0xFF == ord("e"):        #close
                print('Stop collecting close.')
                flag = 2
            elif key & 0xFF == ord("q"):
                print('quit')
                break

            if flag == 1:       #open
                cv2.imwrite('./results/test/0/' + 'right_' + str(time.time()) + '.jpg', right)
                cv2.imwrite('./results/test/0/' + 'left_' + str(time.time()) + '.jpg', left)
                flag = 0

            if flag == 2:       #close
                cv2.imwrite('./results/test/1/' + 'right_' + str(time.time()) + '.jpg', right)
                cv2.imwrite('./results/test/1/' + 'left_' + str(time.time()) + '.jpg', left)
                flag = 0

            queue_ver.popleft()
            queue_frame.popleft()


def cal_eyes_box(ver):
    left_eye_x = []
    left_eye_y = []
    right_eye_x = []
    right_eye_y = []
    for i in range(36, 42):
        left_eye_x.append(int(ver[0][i]))
        left_eye_y.append(int(ver[1][i]))
    for i in range(42, 48):
        right_eye_x.append(int(ver[0][i]))
        right_eye_y.append(int(ver[1][i]))

    left_box = [[max(left_eye_x)+15, max(left_eye_y)+15], [max(left_eye_x)+15, min(left_eye_y)-15],
                [min(left_eye_x)-15, max(left_eye_y)+15], [min(left_eye_x)-15, min(left_eye_y)-15]]
    right_box = [[max(right_eye_x)+15, max(right_eye_y)+15], [max(right_eye_x)+15, min(right_eye_y)-15],
                [min(right_eye_x)-15, max(right_eye_y)+15], [min(right_eye_x)-15, min(right_eye_y)-15]]

    return left_box,right_box


def draw_box(box, img):
    cv2.line(img, box[0], box[1], (255, 0, 0), 2)
    cv2.line(img, box[2], box[3], (255, 0, 0), 2)
    cv2.line(img, box[0], box[2], (255, 0, 0), 2)
    cv2.line(img, box[1], box[3], (255, 0, 0), 2)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb05_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
