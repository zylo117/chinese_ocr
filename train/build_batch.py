import os

chars = open('./char_std_5990.txt', 'r').readlines()
chars = [c.strip() for c in chars]
train = open('../datasets/license_plate_ocr/train.txt', 'r').readlines()
train = [t.strip() for t in train]

line = []

new_train = open('../datasets/license_plate_ocr/new_train.txt', 'w')
for t in train:
    data = t.split(':')
    char_idx_list = []

    for c in data[1]:
        char_idx_list.append(str(chars.index(c)))

    content = data[0] + ' ' + ' '.join(char_idx_list) + '\n'
    new_train.write(content)

new_train.close()
