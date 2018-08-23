# -*- coding:utf-8 -*-
from math import degrees, atan2, fabs, sin, radians, cos

import cv2
import os
import time
import shutil
from glob import glob
import numpy as np
import tensorflow as tf

# init session
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
# sess = tf.Session(config=config)

from network import keys, densenet
from keras.layers import Input
from keras.models import Model


class Config:
    SCALE = 900  # 600
    MAX_SCALE = 1500  # 1200
    TEXT_PROPOSALS_WIDTH = 0  # 16
    MIN_NUM_PROPOSALS = 0  # 2
    MIN_RATIO = 0.01  # 0.5
    LINE_MIN_SCORE = 0.6  # 0.9
    MAX_HORIZONTAL_GAP = 30  # 50
    TEXT_PROPOSALS_MIN_SCORE = 0.7  # 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3  # 0.2
    TEXT_LINE_NMS_THRESH = 0.3
    MIN_V_OVERLAPS = 0.6  # 0.7
    MIN_SIZE_SIM = 0.6  # 0.7


characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

input = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.getcwd(), '/home/admin/github/chinese_ocr_keras/train/models/sheet_vin/weights-densenet-93-6.72.h5')
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


def predict(img):
    height, width = img.shape[:2]
    scale = height * 1.0 / 32
    width = int(width / scale)

    # img = img.resize([width, 32], Image.ANTIALIAS)
    img = cv2.resize(img, (width, 32), interpolation=cv2.INTER_LANCZOS4)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut


def charRec(img, text_recs, adjust=False):
    """
    加载OCR模型，进行字符识别
    """
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]

    for index, rec in enumerate(text_recs):
        # xlength = int((rec[6] - rec[0]) * 0.1)
        # ylength = int((rec[7] - rec[1]) * 0.2)
        # if adjust:
        #     pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
        #     pt2 = (rec[2], rec[3])
        #     pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
        #     pt4 = (rec[4], rec[5])
        # else:
        #     pt1 = (max(1, rec[0]), max(1, rec[1]))
        #     pt2 = (rec[2], rec[3])
        #     pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
        #     pt4 = (rec[4], rec[5])

        # degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度
        #
        # partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
        #
        # if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
        #     continue

        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        text = predict(image)

        if len(text) > 0:
            results[index] = [rec]
            results[index].append(text)  # 识别文字

    return results


if __name__ == '__main__':
    # define test images dir
    # image_files = glob('./datasets/plate/*.jpg')
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    train_set = open('./datasets/license_plate_ocr/new_train.txt').readlines()
    train_set = [d.split(' ')[0] for d in train_set]
    test_set = open('./datasets/license_plate_ocr/new_val.txt').readlines()
    test_set = [d.split(' ')[0] for d in test_set]

    IMAGE_PATH = './datasets/plate'
    # loop over all images
    # for image_file in sorted(image_files):
    for p in os.listdir(IMAGE_PATH):
        stat = ''
        if p in train_set:
            stat = 'train'
        elif p in test_set:
            stat = 'test'
        else:
            continue

        image_file = IMAGE_PATH + '/' + p
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (900, 300), interpolation=cv2.INTER_LANCZOS4)
        t = time.time()

        # arrange box from top to bottom, from left to right
        img, scale = resize_im(img, scale=Config.SCALE, max_scale=Config.MAX_SCALE)
        h, w = img.shape[:2]
        text_recs = [np.array([0, 0, w, 0, 0, h, w, h])]
        # load OCR model and recognize
        result = charRec(img, text_recs, False)

        gt = image_file.split('/')[-1].split('_')[0]
        place = image_file.split('/')[-1].split('_')[-1].strip('.jpg')

        count = 0
        for i, (g, r) in enumerate(zip(gt, result[0][1])):
            if g == r:
                count += 1

        print('{},{},{},{},{},{}'.format(stat, place, gt, result[0][1], count, time.time() - t))
        # # # output result
        # print("Mission complete, it took {:.3f}s".format(time.time() - t))
        # print("\n{} Recognition Result:\n".format(image_file.split('/')[-1]))
        # for key in result:
        #     print(result[key][1])
