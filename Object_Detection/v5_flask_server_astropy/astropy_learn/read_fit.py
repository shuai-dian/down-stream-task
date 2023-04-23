import astropy.io.fits as pyfits
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
import numpy as np
# import sep_use.img_scale as img_scale
import matplotlib
from PIL import Image
import time
# hdulist = pyfits.open('../data/20201220013535730_45696.fit')
# hdulist = pyfits.open('./sat_00000.0101.fits')
# infos = hdulist.info()
# img_data = hdulist[0].data
# # img_data[10,10] = 35535
#
# vmin =  np.min(img_data)
# vmax = np.max(img_data)
# img_data[img_data > vmax] = vmax
# img_data[img_data < vmin] = vmin
# img_data = (img_data - vmin)/(vmax - vmin)
# img_data = (255 * img_data).astype(np.uint8)
# img_data = img_data[::-1,:]
# img_data = Image.fromarray(img_data,'L')
# print(img_data)
#
# image_data = fits.open('./sat_00000.0101.fits')
# print (image_data.info())
# dataImage = image_data[0].data
# print(dataImage)
#
#
# im = Image.fromarray(dataImage)
# # if im.mode != 'RGB':
# im = im.convert('RGB')
# im.save("0101_rgb")
# im.close()

# hdulist = pyfits.open('./sat_00000.0101.fits')
# sig_fract = 5.0
# percent_fract = 0.01


# import numpy
# import pyfits
# import img_scale
# import pylab
# hdulist = pyfits.open('./sat_00000.0101.fits')
# img_header = hdulist[0].header
# img_data_raw = hdulist[0].data
# hdulist.close()
# width=img_data_raw.shape[0]
# height=img_data_raw.shape[1]
# img_data_raw = np.array(img_data_raw, dtype=float)
# #sky, num_iter = img_scale.sky_median_sig_clip(img_data, sig_fract, percent_fract, max_iter=100)
# sky, num_iter = img_scale.sky_mean_sig_clip(img_data_raw, sig_fract, percent_fract, max_iter=10)
# img_data = img_data_raw - sky
# min_val = 0.0
# new_img = img_scale.sqrt(img_data, scale_min = min_val)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('sqrt.png')
# pylab.clf()
# new_img = img_scale.power(img_data, power_index=3.0, scale_min = min_val)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('power.png')
# pylab.clf()
# new_img = img_scale.log(img_data, scale_min = min_val)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('log.png')
# pylab.clf()
# new_img = img_scale.linear(img_data, scale_min = min_val)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('linear.png')
# pylab.clf()
# new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=0.01)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('asinh_beta_01.png')
# pylab.clf()
# new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=0.5)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('asinh_beta_05.png')
# pylab.clf()
# new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=2.0)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('asinh_beta_20.png')
# pylab.clf()
# new_img = img_scale.histeq(img_data_raw, num_bins=256)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('histeq.png')
# pylab.clf()
# new_img = img_scale.logistic(img_data_raw, center = 0.03, slope = 0.3)
# pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
# pylab.axis('off')
# pylab.savefig('logistic.png')
# pylab.clf()



# img_data.save("sat0101.png")
#
# import os
# import numpy as np
# import cv2
#
# path = './.../.../'
# img_name = 'xx.jpg'
# image = cv2.imread(os.path.join(path,img_name)) # 读取图像 path为存储图像路径，img_name为图像文件名
# # 添加噪声
# noise_type = np.random.poisson(lam=0.03,size=(2177,2233,1)).astype(dtype='uint8') # lam>=0 值越小，噪声频率就越少，size为图像尺寸
# noise_image = noise_type+image  # 将原图与噪声叠加
# cv2.imshow('添加噪声后的图像',noise_image)
# cv2.waitKey(0)
# cv2.destroyWindow()




# img_data.save("0900.png")

# plt.imshow(img_data)
# plt.colorbar()
# plt.show()
# hdulist.close()

'''https://docs.astropy.org/en/stable/io/fits/api/images.html#imagehdu
'''
# from astropy.utils.data import get_pkg_data_filename
# from astropy.io import fits
# image_file = get_pkg_data_filename(r'D:\code\guangdian\data\20201220013535730_45696.fit')
# fits.info(image_file)
# # 读取fits图像数据（2D numpy 数组）
# image_data = fits.getdata(image_file)
# # 打印维度
# print(image_data.shape)


def median_filter(input_image, kernel, stride=1, padding=False):
    """
    中值滤波/最大滤波/均值滤波
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return:
    """

    # 填充（默认为1）
    padding_num = 1
    if padding:
        padding_num = int((kernel.shape[0] - 1) / 2)
        input_image = np.pad(input_image, (padding_num, padding_num), mode="constant", constant_values=0)

    out_image = np.copy(input_image)

    # 填充后的图像大小
    w, h = input_image.shape
    print(input_image.shape, padding_num)

    for i in range(padding_num, w - padding_num, stride):
        for j in range(padding_num, h - padding_num, stride):
            region = input_image[i - padding_num:i + padding_num + 1, j - padding_num:j + padding_num + 1]
            # 确保 图像提取的局部区域 与 核大小 一致
            assert (region.shape == kernel.shape)
            # 中值滤波np.median，  最大值滤波 np.maximum  均值滤波： np.mean
            out_image[i, j] = np.median(np.dot(region, kernel))

    # 裁剪原图像大小
    if padding:
        out_image = out_image[padding_num:w - padding_num, padding_num:h - padding_num]
    return out_image

hdulist = pyfits.open('./sat_00000.0101.fits')
infos = hdulist.info()
img_data = hdulist[0].data
print(img_data)
print(type(img_data))
t1 = time.time()
#标准正态分布
kernel = np.random.rand(3, 3)
img_data = median_filter(img_data, kernel)
print(img_data)
t2 = time.time()
print(t2 - t1)










# vmin =  np.min(img_data)
# vmax = np.max(img_data)
# img_data[img_data > vmax] = vmax
# img_data[img_data < vmin] = vmin
# img_data = (img_data - vmin)/(vmax - vmin)
# img_data = (255 * img_data).astype(np.uint8)
# img_data = img_data[::-1,:]
# img_data = Image.fromarray(img_data,'L')
# print(img_data)
# img_data.save("max_lv.png")





# img_data[10,10] = 35535
#
