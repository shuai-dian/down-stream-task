import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import cv2
import pandas as pd
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import convolve,Gaussian2DKernel
hdulist = pyfits.open(r"F:\ali_cloud\download/sat_00000.0185.fits")
image_data = hdulist[0].data.T
print(hdulist[0].data)
header =  hdulist[0].header
print(header)
for i in header:
    print(i,header[i])
print(header["DATE-OBS"])

# from fits_tools import fits_image_reader
# image_data = fits_image_reader('image.fits')
# cv2.imshow("f",image_data)
# cv2.waitKey(0)
hdulist.close()

# image_file = get_pkg_data_filename(file_name)
# image_data = fits.getdata(image_file, ext=0)
# n, bins, patches = plt.hist(image_data.flatten(), 2560)
n, bins, patches = plt.hist(image_data.flatten(), 1400)
bin_dict = {}
for v, bin in zip(n, bins):
    bin_dict[bin] = v
sort_bin = sorted(bin_dict.items(), key=lambda x: x[1], reverse=True)
Sum = sum(n)
tmp_sum = 0
key_list = []
for pair in sort_bin:
    v = pair[1]
    if tmp_sum / Sum < 0.99:
        tmp_sum += v
        key_list.append(pair[0])
    elif v > 10:
        key_list.append(pair[0])

mean_key = np.mean(key_list)
std_key = np.std(key_list)
upper_bound = mean_key + 3 * std_key

final_key = []
for k in key_list:
    if k >= upper_bound:
        pass
    else:
        final_key.append(k)
sort_key = sorted(final_key)
vmin = math.ceil(sort_key[1])
# index = -1
# for i,c in enumerate(n):
#     if c < 5:
#         index = i
#         break
vmax = math.floor(sort_key[-1])
image_data[image_data > vmax] = vmax
image_data[image_data < vmin] = vmin
img_data = (image_data - vmin) / (vmax - vmin)
img_data = (255 * img_data).astype(np.uint8)
kernel = np.array([[0, 1, 0],
                   [1, 3, 1],
                   [0, 1, 0]], dtype=np.uint8)
# kernel2 = np.array([[1, 0, 1],
#                    [0, 1, 0],
#                    [1, 0, 1]], dtype=np.uint8)
# kernel = np.ones((3,3),dtype=np.uint8)
# dilate = cv2.dilate(img_data, kernel, 1)
# img_data = Image.fromarray(img_data,)
# img_data.save("33.jpg")
# plt.imshow(img_data)
# plt.show()
# img_data =  cv2.flip(img_data, -1)
# img_data =  cv2.flip(img_data, 0)/

# cv2.imshow("f",img_data)
# cv2.imwrite("01.png",img_data)
# cv2.waitKey(0)

# # plt.imshow(img_data)
# # plt.show()
# # infos = hdulist.info()
# # print(infos)
# # header = hdulist[0].header
# # print("header",header)
#
# vmin = 0
# vmax = np.max(img_data)
# img_data[img_data > vmax] = vmax
# img_data[img_data < vmin] = vmin
# img_data = (img_data - vmin)/(vmax - vmin)
# img_data = (255 * img_data).astype(np.uint8)
# img_data = img_data[::-1,:]
# img_data = Image.fromarray(img_data,'L')
# # img_data.save("33.jpg")
#
#
# w,h = img_data.shape
# img_data2 = np.zeros((w,h))
# print(img_data2)
# CiShu = 1.5
# Xishu  = 1
# for a1 in range(len(img_data)):
#     for a2 in range(len(img_data[0])):
#         img_data2[a1,a2] = (img_data[a1,a2] ** CiShu) * Xishu

# plt.imshow(img_data2)
# plt.show()

# Gaussian2DKernel1  = Gaussian2DKernel(5)
# img2 = convolve(img_data,Gaussian2DKernel1)
# plt.imshow(img2)
# plt.show()

# print("=====================")
# # img_data.save("1.jpg")

# data2 = pyfits.getdata('sat_00000.0101.fits')
# print(data2)
# print(type(data2))
# print(data2.shape)
# plt.imshow(data2,cmap="gray")
# plt.colorbar()

# vmin = 0
# vmax = np.max(img_data)
# img_data[img_data > vmax] = vmax
# img_data[img_data < vmin] = vmin
# img_data = (img_data - vmin)/(vmax - vmin)
# img_data = (255 * img_data).astype(np.uint8)
# img_data = img_data[::-1,:]
# img_data = Image.fromarray(img_data,'L')
# img_data.save("2.jpg")




# import base64
# with open("sat_00000.0101.fits", 'rb') as image_file:
#     data = base64.b64encode(image_file.read()).decode("utf-8")
#     print(data)
#     f = base64.b64decode(data)
#     f.save("2.jpg")
#

# '''
# img_data = Image(img_data)
# img_dat
# plt.figure()
# plt.imshow(img_data)
# plt.colorbar()
# plt.show()
# hdulist.close()
# from astropy.utils.data import get_pkg_data_filename
# from astropy.io import fits
# image_file = get_pkg_data_filename(r'D:\code\guangdian\data\20201220013535730_45696.fit')
# fits.info(image_file)
# # 读取fits图像数据（2D numpy 数组）
# image_data = fits.getdata(image_file)
# # 打印维度
# print(image_data.shape)


# a ="sat_00000.0101.fits"
# print(a.endswith(".fits"))
