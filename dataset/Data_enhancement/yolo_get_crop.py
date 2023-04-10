import cv2
import numpy as np

import glob
import os
# 读取YOLO标签文件
def read_yolo_label(label_file_path):
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        objects = []
        for line in lines:
            object_info = line.split()
            objects.append(object_info)
    return objects

# 根据标签文件裁剪物体并保存为新的图像
def crop_objects(image_path, label_path):
    image = cv2.imread(image_path)
    objects = read_yolo_label(label_path)
    for obj in objects:
        # 提取物体的位置信息
        label = obj[0]
        x, y, w, h = map(float, obj[1:])
        x1, y1 = int((x - w/2)*image.shape[1]), int((y - h/2)*image.shape[0])
        x2, y2 = int((x + w/2)*image.shape[1]), int((y + h/2)*image.shape[0])
        # 裁剪物体并保存为新的图像
        object_img = image[y1:y2, x1:x2, :]
        object_filename = f"ALL_Ther/crop_img/{label}_{x1}_{y1}_{x2}_{y2}.jpg"
        try:
            cv2.imwrite(object_filename, object_img)
        except Exception as e:
            pass

# 读取每个图像的标签文件并裁剪物体
def process_dataset(image_dir, label_dir):
    image_paths = glob.glob(f"{image_dir}/*.jpg")
    # print(len(image_paths))
    for image_path in image_paths:
        label_path = f"{label_dir}/{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        print(label_path)
        crop_objects(image_path, label_path)

image_dir = "ALL_Ther/train/images"
label_dir = "ALL_Ther/train/labels"
process_dataset(image_dir,label_dir)