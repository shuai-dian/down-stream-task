import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import convolve,Gaussian2DKernel

hdulist = pyfits.open("sat_00000.0101.fits")
print(hdulist[0])
img_data = hdulist[0].data
w,h = img_data.shape
print(img_data)
# vmax = np.max(img_data)
# vmin = np.min(img_data)
# print("max",vmax,"vmin",vmin)
# img_data[img_data > vmax] = vmax
# img_data[img_data < vmin] = vmin
# img_data = (img_data - vmin)/(vmax - vmin)
# img_data = (255 * img_data).astype(np.uint8)
# img_data = Image.fromarray(img_data)
# if img_data.mode != 'RGB':
#     img_data = img_data.convert('RGB')
# img_data.save("44.jpg")
#


# plt.imshow(img_data)
# plt.show()




###### 灰度量级淡化弱点
# leave_gray = 2
# liangji= np.max(img_data) / leave_gray
# img2 = np.zeros((w,h))
# for a1 in range(w):
#     for a2 in range(h):
#         img2[a1,a2] = int(img_data[a1,a2] / liangji)
#
# vmin = 0
# vmax = np.max(img2)
# img2[img2 > vmax] = vmax
# img2[img2 < vmin] = vmin
# img2 = (img2 - vmin)/(vmax - vmin)
# img2 = (255 * img2).astype(np.uint8)
# img2 = Image.fromarray(img2)
# if img2.mode != 'RGB':
#     img2 = img2.convert('RGB')
# img2.save("44.jpg")
#################
###################对比度 ，亮度调整与幂次变换
# img2 = np.zeros((w,h))
# cishu = 1.5
# xishu = 1
# for a1 in range(w):
#     for a2 in range(h):
#         img2[a1,a2] = (img_data[a1,a2]** cishu) * xishu
# vmin = 0
# vmax = np.max(img2)
# img2[img2 > vmax] = vmax
# img2[img2 < vmin] = vmin
# img2 = (img2 - vmin)/(vmax - vmin)
# img2 = (255 * img2).astype(np.uint8)
# img2 = Image.fromarray(img2)
# if img2.mode != 'RGB':
#     img2 = img2.convert('RGB')
# img2.save("幂次增强.jpg")
# #################



##############高斯放大亮度范围
# Gaussian2DKernel1  = Gaussian2DKernel(3)
# img2 = convolve(img_data,Gaussian2DKernel1)
#
# vmin = 0
# vmax = np.max(img2)
# img2[img2 > vmax] = vmax
# img2[img2 < vmin] = vmin
# img2 = (img2 - vmin)/(vmax - vmin)
# img2 = (255 * img2).astype(np.uint8)
# img2 = Image.fromarray(img2)
# if img2.mode != 'RGB':
#     img2 = img2.convert('RGB')
# img2.save("高斯模糊.jpg")

# img_data = Image.fromarray(img_data,'L')
# img_data.save("3.jpg")
# plt.imshow(img/2)
# plt.show()