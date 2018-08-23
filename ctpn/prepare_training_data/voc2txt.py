import os
import xmltodict

IMAGE_PATH = '../datasets/test/image'
LABEL_PATH = '../datasets/test/raw_label'
GT_PATH = '../datasets/test/label'

ipl = os.listdir(IMAGE_PATH)

for ip in ipl:
    if ip.endswith('.jpg'):
        base_name = ip.rstrip('.jpg')
        print(base_name)
        gt = xmltodict.parse(open(LABEL_PATH + '/' + base_name + '.xml', encoding='gbk').read())
        filename = gt['annotation']['filename']
        obj = gt['annotation']['object']

        lines = []
        for o in obj:
            bnd_box = o['bndbox']
            line = ','.join([bnd_box['xmin'], bnd_box['ymin'],
                             bnd_box['xmax'], bnd_box['ymin'],
                             bnd_box['xmax'], bnd_box['ymax'],
                             bnd_box['xmin'], bnd_box['ymax'],
                             'sheet_zh_cn', '\n'])
            lines.append(line)

        with open(GT_PATH + '/' + base_name + '.txt', 'w') as f:
            f.writelines(lines)
            f.close()
