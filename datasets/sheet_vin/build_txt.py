import os
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

char = open('../../train/char_std_5990.txt').readlines()
char = [c.strip() for c in char]

pl = glob('images/*.jpg')
lines = []
max_label_len = 0
for p in pl:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    if w / h < 280 / 32:  # make sure all image scale is smaller than 280 / 32
        basename = os.path.basename(p)
        gt = basename.strip('.jpg').split('_')[0]
        idx = [basename]

        if len(gt) > max_label_len:
            max_label_len = len(gt)

        for s in gt:
            idx.append(str(char.index(s)))

        lines.append(' '.join(idx) + '\r')

train_data, val_data = train_test_split(lines, test_size=0.05, random_state=42)
open('train.txt', 'w').writelines(train_data)
open('val.txt', 'w').writelines(val_data)

print(max_label_len)
