import cv2
import os
from PIL import Image
import numpy as np

from collections import Counter

images_dir = "TIR-SS/Img8bit/test"
labels_dir = "TIR-SS/gtFine/test"
img_list = os.listdir(images_dir)
for i in img_list:
    img_cp = os.path.join(images_dir,i)
    lab_cp = os.path.join(labels_dir,i.replace(".bmp",".png"))
    # 打开图像文件和分类标签
    ir_image = Image.open(img_cp)
    label_image = Image.open(lab_cp)
    # 将分类标签映射到颜色
    color_map = {
        0: (0, 0, 0),  # 背景 road
        1: (255, 0, 0),  # 类别1  sidewalk
        2: (0, 255, 0),  # 类别2 pedestrian
        3: (65, 65, 65),  # 类别3 rider
        4: (114, 114, 255),  # 类别3 vehicle
        5: (114, 114, 114),  # 类别3 building
        6: (200, 0, 255),  # 类别3 vegetation
        7: (200, 200, 255),  # 类别3 sky
        8: (255, 255, 255),  # 类别3 background
    }
    label_array = np.array(label_image)
    label_array = label_array.reshape(1,-1).tolist()[0]
    my_counter = Counter(label_array)
    # print(my_counter)
    # # 为每个像素绘制颜色
    # for y in range(image.size[1]):
    #     for x in range(image.size[0]):
    #         label_value = label_array[y, x]
    #         color = color_map[label_value]
    #         image.putpixel((x, y), color)
    #
    # # 保存带有标注的图像
    # image.save("annotated_image.png")
    if 2 in my_counter.keys():
        mask = Image.new('L', label_image.size)
        mask.putdata([255 if p == 2 else 0 for p in label_image.getdata()])
        # 计算指定label的像素的边界框
        bbox = mask.getbbox()
        # 获取目标的宽和高
        target_width = bbox[2] - bbox[0]
        target_height = bbox[3] - bbox[1]
        # mask_box = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
        # mask_box.paste(mask.crop(bbox), (0, 0))
        # mask_path = os.path.join("TIR-SS/person_mask",i.replace(".bmp",".png"))
        # mask_box.save(mask_path)
        # 保存结果图像
        box_img = ir_image.crop(bbox)
        box_mask = mask.crop(bbox)
        masked_image = Image.composite(box_img, Image.new('RGB', box_img.size, 'black'),box_mask )
        result_image = Image.new('RGBA', box_img.size, (0, 0, 0, 0))
        result_image.putdata([(r, g, b, a) if m > 0 else (0, 0, 0, 0) for (r, g, b), m, a in
                              zip(masked_image.getdata(), box_mask.getdata(), [255] * box_mask.size[0] * box_mask.size[1])])
        img_path = os.path.join("TIR-SS/person_img",i.replace(".bmp",".png"))

        result_image.save(img_path)
