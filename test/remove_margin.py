import os
import numpy as np

import cv2

IMAGE_PATH = 'E:/Document/WXWork/1688852623384721/Cache/File/2018-08/result'
PROCESSED = 'E:/Document/WXWork/1688852623384721/Cache/File/2018-08/result_p'

for p in os.listdir(IMAGE_PATH):
    if p.endswith('.jpg'):
        img_path = IMAGE_PATH + '/' + p
        ori_img = cv2.imread(img_path)
        h, w = ori_img.shape[:2]
        # cv2.imshow('image', ori_img)

        img = cv2.medianBlur(ori_img, 5)
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # cv2.imshow('thresh', thresh)

        thresh = cv2.copyMakeBorder(thresh, h // 4, h // 4, w // 4, w // 4,
                                    cv2.BORDER_CONSTANT, value=0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        closed = cv2.dilate(closed, None, iterations=4)
        closed = closed[h // 4: -h // 4, w // 4:-w // 4]
        # cv2.imshow('closed', closed)

        cnts = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts) > 0:
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            c_tmp = c[:,0,:]
            min = np.min(c_tmp, axis=0)
            max = np.max(c_tmp, axis=0)
            xmax, ymax = max[0],max[1]
            xmin, ymin = min[0],min[1]

            roi = ori_img[ymin:ymax,xmin:xmax]
            # cv2.imshow('roi', roi)

            # cv2.waitKey(0)

            cv2.imwrite(PROCESSED + '/' + p, roi, [cv2.IMWRITE_JPEG_QUALITY, 100])
