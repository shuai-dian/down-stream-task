from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter
import astropy.io.fits as pyfits
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import time
import cv2
from scipy import ndimage

def median_filter(input_image,kernel,stride=1,padding=False):
    padding_num = 1
    if padding:
        padding_num = int((kernel.shape[0]-1)/2)
        input_image = np.pad(input_image,(padding_num,padding_num),mode="constant",constant_values=0)
    out_image = np.copy(input_image)
    # 填充后的图像大小
    w,h = input_image.shape
    for i in range(padding_num,w-padding_num,stride):
        for j in range(padding_num,h-padding_num,stride):
            region = input_image[i-padding_num:i+padding_num+1,j-padding_num:j+padding_num+1]
            print(region)
            # 确保 图像提取的局部区域 与 核大小 一致
            assert (region.shape == kernel.shape)
            # 中值滤波np.median，  最大值滤波 np.maximum  均值滤波： np.mean
            out_image[i,j] = np.maximum(np.dot(region,kernel))
            # out_image[i,j] = np.median(np.dot(region,kernel))
    # 裁剪原图像大小
    if padding:
        out_image = out_image[padding_num:w-padding_num,padding_num:h-padding_num]
    return out_image


def max_median(image_data,kernel,stride=1,padding=None):
    w, h = image_data.shape
    if not padding:  # 当没有边缘时
        print("padding")
        edge = int((kernel - 1) / 2)  # 边缘忽略值
        print(edge)
        if h - 1 - edge <= edge or w - 1 - edge <= edge: # 512-1-1<=1 or 512-1-1 < 1
            print("The parameter k is to large.")
            return None

        new_arr = np.zeros((w, h), dtype="uint8")
        # new_arr = np.zeros((w, h))
        for i in range(w):
            for j in range(h):
                if i <= edge - 1 or i >= h - 1 - edge or j <= edge - 1 or j >= h - edge - 1:
                    new_arr[i, j] = image_data[i, j]
                else:  # 没有设计排序算法，直接使用Numpy中的寻找中值函数
                    Z1 = np.median(np.array([img_data[i-1,j],img_data[i,j],img_data[i+1,j]]))
                    Z2 = np.median(np.array([img_data[i,j-1],img_data[i,j],img_data[i,j+1]]))
                    Z3 = np.median(np.array([img_data[i-1,j-1],img_data[i,j],img_data[i+1,j+1]]))
                    Z4 = np.median(np.array([img_data[i+1,j-1],img_data[i,j],img_data[i-1,j+1]]))
                    new_arr[i, j] = int(max([Z1,Z2,Z3,Z4]))
                    # print(max([Z1,Z2,Z3,Z4]))
                    # print(image_data[i - edge:i + edge + 1, j - edge:j + edge + 1])
                    # new_arr[i, j] = np.median(image_data[i - edge:i + edge + 1, j - edge:j + edge + 1])
        print(new_arr)
    return new_arr


hdulist = pyfits.open('./sat_00000.0101.fits')
# hdulist = pyfits.open('./gimg-0900.fits')
infos = hdulist.info()
img_data = hdulist[0].data
print(img_data)
w,h = img_data.shape
kernel = 3

# t1 = time.time()
img_data = max_median(img_data, kernel)
# print(time.time() - t1)

### cv median filter
def cv_max_median(img_data):
    img_data = cv2.medianBlur(img_data, 3)
    w,h = img_data.shape
    kernel = 3
    edge = int((kernel - 1) / 2)  # 边缘忽略值
    new_arr = np.zeros((w, h), dtype="uint8")
    for i in range(w):
        for j in range(h):
            if i <= edge - 1 or i >= h - 1 - edge or j <= edge - 1 or j >= h - edge - 1:
                new_arr[i, j] = img_data[i, j]
            else:  # 没有设计排序算法，直接使用Numpy中的寻找中值函数
                Z1 = [img_data[i - 1, j], img_data[i, j], img_data[i + 1, j],img_data[i, j - 1], img_data[i, j], img_data[i, j + 1],img_data[i - 1, j - 1], img_data[i, j], img_data[i + 1, j + 1],img_data[i + 1, j - 1], img_data[i, j], img_data[i - 1, j + 1]]
                # Z2 = np.array([img_data[i, j - 1], img_data[i, j], img_data[i, j + 1]])
                # Z3 = np.array([img_data[i - 1, j - 1], img_data[i, j], img_data[i + 1, j + 1]])
                # Z4 = np.array([img_data[i + 1, j - 1], img_data[i, j], img_data[i - 1, j + 1]])
                new_arr[i, j] = max(Z1)
    return new_arr
### cv median filter
def cv_max_median2(img_data):
    src1 = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0],
    ])
    src2 = np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0],
    ])
    src3 = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ])
    src4 = np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0],
    ])
    img_data1 = cv2.filter2D(img_data,-1,src1)/3
    img_data2 = cv2.filter2D(img_data,-1,src2)/3
    img_data3 = cv2.filter2D(img_data,-1,src3)/3
    img_data4 = cv2.filter2D(img_data,-1,src4)/3
    w,h = img_data.shape
    new_arr = np.zeros((w, h), dtype="uint8")
    for i in range(w):
        for j in range(h):
            new_arr[i, j] =max(img_data1[i,j],img_data2[i,j],img_data3[i,j],img_data4[i,j])
    print(new_arr)
    return new_arr
# img_data = cv_max_median2(img_data)
#
# # vmin =  np.min(img_data)
# # vmax = np.max(img_data)
# # img_data[img_data > vmax] = vmax
# # img_data[img_data < vmin] = vmin
# # img_data = (img_data - vmin)/(vmax - vmin)
# # img_data = (255 * img_data).astype(np.uint8)
# img_data = img_data[::-1,:]
# img_data = Image.fromarray(img_data)
# img_data.save("cvmax_median2.png")


### max median cv
# kernel1 = np.array([
#     [0,1,0],
#     [0,1,0],
#     [0,1,0],
# ])
# kernel2 = np.array([
#     [0,0,0],
#     [1,1,1],
#     [0,0,0],
# ])
# resulting_image = cv2.filter2D(img_data, -1, kernel)
# print(resulting_image)


# vmin =  np.min(img_data)
# vmax = np.max(img_data)
# img_data[img_data > vmax] = vmax
# img_data[img_data < vmin] = vmin
# img_data = (img_data - vmin)/(vmax - vmin)
# img_data = (255 * img_data).astype(np.uint8)
img_data = img_data[::-1,:]
img_data = Image.fromarray(img_data)
img_data.save("max_median2.png")




