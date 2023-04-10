import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import shutil
import cv2
import xml.etree.ElementTree as ET
import os

VOC_NAMES = ['car', 'person']

def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


if __name__ == '__main__':
    anno_dir = "valid/anno"
    images_dir = "valid/images"
    lab_list = ['car', 'person','bus']
    for i in os.listdir(images_dir):
        img_cp = os.path.join(images_dir, i)
        anno_cp = os.path.join(anno_dir, i.replace(".jpg",".xml"))
        yolo_cp = os.path.join("valid/labels", i.replace(".jpg", ".txt"))
        # image = cv2.imread(img_cp)

        tree = ET.parse(anno_cp)
        root = tree.getroot()
        out_file = open(yolo_cp, 'w')

        for obj in root.iter('object'):
            # 获取框的坐标信息
            names = obj.find("name").text # ['car', 'person', 'motorcycle', 'bicycle', 'dog', 'traffic light', 'bus']
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if names in lab_list:
                if names == "bus":
                    names = "car"
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                bb = convert_box((w, h),[xmin,xmax,ymin,ymax] )

                cls_id = VOC_NAMES.index(names)  # class id

                out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

            # if names in lab_list:
            #     pass
            # else:
            #     lab_list.append(names)

            # 绘制框在图像上
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 保存画框后的图像
        # cv2.imwrite(save_dir,image)
    # print(lab_list)
    # break
# break
# print(lab_list)

