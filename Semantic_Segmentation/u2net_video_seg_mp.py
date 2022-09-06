import os
import torch
from torch.autograd import Variable
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image

from model import U2NET # full size version 173.6 MB

import cv2
import time
import torch.multiprocessing as mp

def get_frame(cap):        #可能需要放到主程序里
    ret, frame = cap.read()
    return frame

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def naive_cutout(img, mask):    #uint8转float32?
    mask = mask * 255
    mask = mask.astype(np.uint8)
    cv2.imshow("mask", mask)
    #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cutout = cv2.bitwise_and(img,img,mask=mask)

    return cutout

def predict(q1,q2):
    cap = cv2.VideoCapture(0)

    model_name = 'u2net'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + '_human_seg', model_name + '_human_seg.pth')

    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    while True:
        with torch.no_grad():

            start = time.time()

            input = get_frame(cap)
            copy = input
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = Image.fromarray(input, "RGB")
            input = transforms.ToTensor()(input)
            input = input.type(torch.FloatTensor)
            input = input.unsqueeze(0)
            input = Variable(input.cuda())

            if torch.cuda.is_available():
                input = Variable(input.cuda())
            else:
                input = Variable(input)

            d1, d2, d3, d4, d5, d6, d7 = net(input)

            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            end = time.time()
            print("模型时间为：" + str(end - start))

            q1.put(copy)
            #pred.cpu()耗时极长
            q2.put(pred.cpu())

            end_2 = time.time()
            print("转换时间为：" + str(end_2 - end))

            torch.cuda.empty_cache()

            if q1.qsize() > 4:
                q1.get()

            if q2.qsize() > 4:
                q2.get()

def segmentation(q1,q2):
    rec = time.time()
    pro_end = rec
    while True:

        predict = q2.get()
        input = q1.get()

        pro_start = time.time()
        print("间隔时间为:" + str(pro_start - pro_end))

        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        start = time.time()
        result = naive_cutout(input, predict_np)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        end = time.time()
        print("切割时间为：" + str(end - start))

        print("帧时间为：" + str(time.time() - rec))
        rec = time.time()
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pro_end = time.time()

if __name__ == '__main__':
    mp.set_start_method(method='spawn')

    q1 = mp.Queue(maxsize=5)    #origin image
    q2 = mp.Queue(maxsize=5)    #mask

    process1 = mp.Process(target=predict, args=(q1,q2))
    process2 = mp.Process(target=segmentation, args=(q1,q2))
    process1.daemon = True
    process2.daemon = True
    process1.start()
    process2.start()
    process1.join()
    process2.join()