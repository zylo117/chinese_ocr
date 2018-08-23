# -*- coding:utf-8 -*-
import cv2
import os
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob

if __name__ == '__main__':
    from ctpn.lib.fast_rcnn.config import cfg_from_file

    # load cfg(checkpoints path, RUN_ON devices, GPU_MEN_USAGE, etc)
    cfg_from_file('./ctpn/ctpn/text.yml')

    from ctpn.text_detect import text_detect
    import ocr

    # define test images dir
    image_files = glob('./test_images/2018*.*')
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    # loop over all images
    for image_file in sorted(image_files):
        img = np.array(Image.open(image_file).convert('RGB'))
        t = time.time()

        # detect text area
        text_recs, img_framed, img = text_detect(img)
        # arrange box from top to bottom, from left to right
        text_recs = ocr.sort_box(text_recs)

        # suck it, amateur
        # p1 - p2
        # p3 - p4
        # to
        # p1 - p2
        # p4 - p3
        for i, tr in enumerate(text_recs):
            pt1, pt2, pt3, pt4 = (tr[0], tr[1]), (tr[2], tr[3]), (tr[6], tr[7]), (tr[4], tr[5])
            # cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            # cv2.line(img, pt2, pt3, (0, 255, 0), 2)
            # cv2.line(img, pt3, pt4, (0, 255, 0), 2)
            # cv2.line(img, pt4, pt1, (0, 255, 0), 2)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            min_x = min([pt1[0], pt4[0]])
            max_x = max([pt2[0], pt3[0]])
            min_y = min([pt1[1], pt2[1]])
            max_y = max([pt3[1], pt4[1]])
            w = max_x - min_x
            h = max_y - min_y

            if w > 0 and h > 0:
                roi = img[min_y:max_y, min_x:max_x]
                old_M = np.array([[pt1[0] - min_x, pt1[1] - min_y], [pt2[0] - min_x, pt2[1] - min_y],
                                  [pt3[0] - min_x, pt3[1] - min_y], [pt4[0] - min_x, pt4[1] - min_y]]).astype(
                    np.float32)
                new_M = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)

                M = cv2.getPerspectiveTransform(old_M, new_M)
                roi = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

                # cv2.imshow('roi', roi)
                # cv2.waitKey(0)
                cv2.imwrite(
                    result_dir + '/{}_{}.jpg'.format(image_file.split(os.path.sep)[-1].strip('.jpg'), i), roi,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])

        # output result
