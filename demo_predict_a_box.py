# -*- coding:utf-8 -*-
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
    image_files = glob('./test_images/02*.*')
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
        # load OCR model and recognize
        result = ocr.charRec(img, text_recs, False)

        # output result
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(img_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
