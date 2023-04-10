import os
import imagehash
from PIL import Image

# 图像文件夹路径
img_dir = 'ALL_Ther/crop_img'

# 获取所有图像的pHash值
hashes = {}
for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    if os.path.isfile(img_path):
        with Image.open(img_path) as img:
            hash_val = imagehash.phash(img)
            hashes[img_path] = hash_val

# 找到所有相似的图像
similar_imgs = {}
for img_path, hash_val in hashes.items():
    similar_imgs.setdefault(str(hash_val), []).append(img_path)

# 删除所有相似图像中除第一张之外的图像
for hash_val, img_list in similar_imgs.items():
    if len(img_list) > 1:
        for img_path in img_list[1:]:
            os.remove(img_path)
