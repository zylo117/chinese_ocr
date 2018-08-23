import os
import shutil
import pandas as pd
import numpy as np

RAW_IMAGES_PATH = '/home/public/car_exam/raw_images/'
GROUND_TRUTH_IMAGES_PATH = '/home/public/car_exam/gt_labeled_images/'
GROUND_TRUTH_CSV = './ground_truth.csv'

df = pd.read_csv(GROUND_TRUTH_CSV, encoding='utf-8')
no = np.asarray(df['作业流水号']).astype(str)
license_plate_no = np.asarray(df['号牌号码']).astype(str)

lp_dict = {}
for i, n in enumerate(no):
    lp_dict[n] = [license_plate_no[i]]

pl = os.listdir(RAW_IMAGES_PATH)
for p in pl:
    if p in no:
        target_path = GROUND_TRUTH_IMAGES_PATH + '/{}_{}'.format(lp_dict[p][0], p)
        if not os.path.exists(target_path):
            shutil.copytree(RAW_IMAGES_PATH + '/' + p, GROUND_TRUTH_IMAGES_PATH + '/{}_{}'.format(lp_dict[p][0], p))
            print(p + ' to ' + lp_dict[p][0])
        else:
            print('skipping existed target path')
