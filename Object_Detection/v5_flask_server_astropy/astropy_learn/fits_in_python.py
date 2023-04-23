from astropy.io import fits
from astropy.table import Table
from collections import Counter
import numpy as np
from numpy import mean,var,std
from scipy import stats
from PIL import Image
# hdu = fits.open('sat_00000.0101.fits', mode='update')[0]
hdu = fits.open('gimg-0900.fits', mode='update')[0]
img_data = hdu.data
unique, counts = np.unique(img_data, return_counts=True)
dict = dict(zip(unique, counts))
min_v = 0
max_v = 65535
mean_v = mean(img_data)
cnt_mean = mean(counts)
cnt_std = std(counts)
print(cnt_mean)
print(cnt_std)

# #显著性水平
# a = 0.05
# # 单测 左分位点
# norm_a_left = stats.norm.ppf(a)
# # 单侧 右分位点
# norm_a_right = stats.norm.isf(a)
# # 双侧分位点
# norm_a_2 = stats.norm.interval(1-a)
#
# print('单侧左分位点：',norm_a_left )
# print('单侧右分位点：',norm_a_right )
# print('双侧分位点：',norm_a_2 )
def norm_conf(data, cnt_mean,cnt_std, confidence=0.9):
    sample_mean = cnt_mean
    sample_size = len(data)
    alpha = 1 - confidence  # 显著性水平
    norm_score = stats.norm.isf(alpha / 2)  # 查表得正态分布的分数
    ME = cnt_std / np.sqrt(sample_size) * norm_score
    lower_limit = sample_mean - ME
    upper_limit = sample_mean + ME
    return lower_limit, upper_limit

# lower_limit, upper_limit = norm_conf(np.array(counts),cnt_mean,cnt_std)
lower_limit, upper_limit  = 150,5
print("lower_limit, upper_limit ",lower_limit, upper_limit)
for i,num in enumerate(counts):
    if num > lower_limit:
        min_v = unique[i]
        break

T_cont = np.flipud(counts)
T_unique = np.flipud(unique)
for i,num in enumerate(T_cont):
    if num > upper_limit:
        max_v = T_unique[i]
        break
print("min_v",min_v,"max_v:",max_v)

vmin = min_v
vmax = max_v
img_data[img_data > vmax] = vmax
img_data[img_data < vmin] = vmin
img_data = (img_data - vmin)/(vmax - vmin)
img_data = (255 * img_data).astype(np.uint8)
img_data = img_data[::-1,:]
img_data = Image.fromarray(img_data,'L')
img_data.save("set_minmax.png")
hdu.close()