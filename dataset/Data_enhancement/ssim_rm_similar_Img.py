import os
import cv2
import numpy as np

# 图像文件夹路径
img_dir = 'ALL_Ther/crop_img'

# 相似度阈值
threshold = 0.9

# 获取所有图像
imgs = []
for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        imgs.append((img_path, img))

# 计算所有图像之间的相似度
similar_imgs = {}
for i in range(len(imgs)):
    for j in range(i + 1, len(imgs)):
        img1_path, img1 = imgs[i]
        img2_path, img2 = imgs[j]
        ssim_val = cv2.compareStructures(img1, img2, cv2.HISTCMP_CORREL)
        if ssim_val > threshold:
            similar_imgs.setdefault(img1_path, []).append(img2_path)
            similar_imgs.setdefault(img2_path, []).append(img1_path)

# 删除所有相似图像中除第一张之外的图像
for img_path, similar_list in similar_imgs.items():
    if len(similar_list) > 0:
        for similar_path in similar_list:
            os.remove(similar_path)
