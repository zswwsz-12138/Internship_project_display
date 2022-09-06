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
import multiprocessing as mp

def get_frame(cap):        #可能需要放到主程序里
    ret, frame = cap.read()
    return frame

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def naive_cutout(img, mask):    #uint8转float32?
    mask = mask[0] * 255
    mask = mask.astype(np.uint8)
    cv2.imshow("mask", mask)
    #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cutout = cv2.bitwise_and(img,img,mask=mask)

    return cutout

def predict(input):
    input= cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = Image.fromarray(input, "RGB")
    input = transforms.ToTensor()(input)
    input = input.type(torch.FloatTensor)
    input = input.unsqueeze(0)

    if torch.cuda.is_available():
        input = Variable(input.cuda())
    else:
        input = Variable(input)

    d1, d2, d3, d4, d5, d6, d7 = net(input)

    # normalization
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    del d1, d2, d3, d4, d5, d6, d7

    return pred

def segmentation(pred, input):
    predict = pred
    predict_np = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    #predict_np = predict.numpy()
    #input = input.numpy()

    #im = Image.fromarray(predict_np * 255).convert('RGB')
    #imo = im.resize((input.shape[1],input.shape[0]), resample=Image.BILINEAR)
    #pb_np = np.array(imo)

    result = naive_cutout(input, predict_np)
    #result = np.asarray(result)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result

if __name__ == '__main__':
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

            input_img = get_frame(cap)

            #input_img.show()
            pred = predict(input_img)

            check = time.time()
            print("模型耗时：" + str(check - start))

            """
            if torch.cuda.is_available():
                input_img = torch.tensor(input_img)
                input_img.cuda()
            """

            result = segmentation(pred,input_img)       #耗时步骤
            stop = time.time()
            print("切割耗时" + str(stop - check))
            cv2.imshow("result",result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            torch.cuda.empty_cache()

            #time.sleep((0.1 - time.time() + begin) if time.time() - begin < 0.1 else 0.0)

    cap.release()
    cv2.destroyAllWindows()