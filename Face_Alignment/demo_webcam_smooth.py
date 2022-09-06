# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
import torch
from PIL import Image
from scipy.spatial import distance as dist

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark
import torchvision.transforms as transforms
import res2net
from utils.pose import my_pose_angle


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    resnet = res2net.__dict__[args.arch]()
    resnet_dict = torch.load(r'.\models\res2net50\model.pth')
    resnet.load_state_dict(resnet_dict)
    resnet.eval()
    # resnet.cuda()

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # resize到224x224大小
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])

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
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(frame_bgr)

            if len(boxes) == 0: # 后期添加，防止报错
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
            # todo: add confidence threshold to judge the tracking is failed 追踪
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

            # 计算并绘制眼部方框
            left_box, right_box = cal_eyes_box(ver)

            img_draw = draw_box(left_box, img_draw)
            img_draw = draw_box(right_box, img_draw)

            # 计算并绘制头部角度
            pose = my_pose_angle(param_lst)
            img_draw = cv2.putText(img_draw, "yaw" + str(round(pose[0], 2)), [350, 50], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            img_draw = cv2.putText(img_draw, "pitch" + str(round(pose[1], 2)), [350, 100], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            img_draw = cv2.putText(img_draw, "roll" + str(round(pose[2], 2)), [350, 150], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # 检测左、右眼睁眼、闭合
            left = frame[left_box[1][1]:left_box[0][1], left_box[2][0]:left_box[0][0]]
            left = Image.fromarray(left)
            left = test_transform(left)
            left = torch.unsqueeze(left, 0)
            left = resnet(left).numpy().squeeze()
            if left[0] > left[1]:
                img_draw = cv2.putText(img_draw, "left-open", [50,50], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            else:
                img_draw = cv2.putText(img_draw, "left-close", [50,50], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            right = frame[right_box[1][1]:right_box[0][1], right_box[2][0]:right_box[0][0]]
            right = Image.fromarray(right)
            right = test_transform(right)
            right = torch.unsqueeze(right, 0)
            right = resnet(right).numpy().squeeze()
            if right[0] > right[1]:
                img_draw = cv2.putText(img_draw, "right-open", [50,100], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            else:
                img_draw = cv2.putText(img_draw, "right-close", [50,100], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # 检测张嘴、闭嘴
            img_draw = mouth_detect(ver, img_draw)

            # cv2.imshow('cut', left)
            cv2.imshow('image', img_draw)
            k = cv2.waitKey(1)
            if (k & 0xff == ord('q')):
                break

            queue_ver.popleft()
            queue_frame.popleft()


def cal_eyes_box(ver):      # 计算眼部方框
    left_eye_x = []
    left_eye_y = []
    right_eye_x = []
    right_eye_y = []
    for i in range(42, 48):
        left_eye_x.append(int(ver[0][i]))
        left_eye_y.append(int(ver[1][i]))
    for i in range(36, 42):
        right_eye_x.append(int(ver[0][i]))
        right_eye_y.append(int(ver[1][i]))

    left_box = [[max(left_eye_x)+12, max(left_eye_y)+12], [max(left_eye_x)+12, min(left_eye_y)-12],
                [min(left_eye_x)-12, max(left_eye_y)+12], [min(left_eye_x)-12, min(left_eye_y)-12]]
    right_box = [[max(right_eye_x)+12, max(right_eye_y)+12], [max(right_eye_x)+12, min(right_eye_y)-12],
                [min(right_eye_x)-12, max(right_eye_y)+12], [min(right_eye_x)-12, min(right_eye_y)-12]]


    return left_box,right_box


def draw_box(box, img):     # 绘制眼部方框
    cv2.line(img, box[0], box[1], (255, 0, 0), 2)
    cv2.line(img, box[2], box[3], (255, 0, 0), 2)
    cv2.line(img, box[0], box[2], (255, 0, 0), 2)
    cv2.line(img, box[1], box[3], (255, 0, 0), 2)

    return img


def mouth_detect(ver, img_draw):        # 计算嘴巴开闭
    out_top = [ver[0][51], ver[1][51]]
    out_bottom = [ver[0][57], ver[1][57]]
    in_top = [ver[0][62], ver[1][62]]
    in_bottom = [ver[0][66], ver[1][66]]

    rate = dist.euclidean(out_top, out_bottom) / dist.euclidean(in_top, in_bottom)
    if rate < 2.3:      # 根据这个比例来判断，可调整数值
        img_draw = cv2.putText(img_draw, "mouth-open", [50, 150], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        img_draw = cv2.putText(img_draw, "mouth-close", [50, 150], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    return img_draw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb05_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=False)
    parser.add_argument('--arch', default='res2net50',
                        choices=res2net.__all__,
                        help='model architecture')

    args = parser.parse_args()
    main(args)
