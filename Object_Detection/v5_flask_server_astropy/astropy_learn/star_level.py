import numpy as np


'''
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

# 导入FITS图像数据
# image_data = fits.getdata('FOCx38i0101t_c0f.fits')
hdulist = fits.open(r"F:\ali_cloud\download\sat_00000.0185.fits")

x_pixel = 512
y_pixel = 512

# 通过FITS图像的头信息创建一个WCS对象
wcs = WCS(hdulist[0].header)

# 将像素坐标转换为天球坐标
ra, dec = wcs.all_pix2world(x_pixel, y_pixel,0)
coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')

v = Vizier(columns=['*', '_RAJ2000', '_DEJ2000'], row_limit=1)
print(v)
result = v.query_region(coords, radius=1 * u.arcsec, catalog='I/259/tyc2')
print(result)
table = result[0]
mag_vt = table['VTmag'][0]
b_v = table['BTmag'][0] - table['VTmag'][0]

mag = mag_vt + 0.09 * b_v
print(mag)
# 创建一个SkyCoord对象，表示该位置的天球坐标
# position = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')

# 使用SkyCoord.match_to_catalog_sky函数，在Tycho-2星表中查找最接近的恒星
#idx, d2d, _ = position.match_to_catalog_sky(tycho2_table)
#
# # 检查匹配距离是否小于一定阈值，以确保匹配的是正确的恒星
# if d2d < 1 * u.arcsec:
#     star = tycho2_table[idx]
# else:
#     print('No star found within 1 arcsecond')

# # 从Tycho-2星表中获取B和V波段的星等
# Bmag = star['BTmag']
# Vmag = star['VTmag']
#
# # 计算B-V的颜色指数
# B_V = Bmag - Vmag
#
# # 使用公式计算绝对星等
# abs_mag = Vmag - 5 * (np.log10(distance) - 1)
#
# # 计算视星等
# app_mag = abs_mag + 5 * np.log10(distance) - 5 + extinction
#
# # 输出结果
# print(f"Apparent magnitude: {app_mag:.2f}")
'''

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

# 导入FITS图像数据
hdulist = fits.open(r"F:\ali_cloud\download\sat_00000.0185.fits")
print("load FITS done")
data = hdulist[0].data

table = Table.read('tycho2.fits', format='fits')
coords = SkyCoord(ra=table['RAdeg'], dec=table['DEdeg'], unit=(u.hourangle, u.deg), frame='icrs')
print(coords)

mags = table['VT']
x_pixel = 512
y_pixel = 512
print("load tycho-2 table done")
# 通过FITS图像的头信息创建一个WCS对象
wcs = WCS(hdulist[0].header)
print(wcs)
# 将像素坐标转换为天球坐标
ra, dec = wcs.all_pix2world(x_pixel, y_pixel,0)
pix_coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
idx, d2d, _ = pix_coords.match_to_catalog_sky(coords)
print("find star done")
pixel_mags = mags[idx]
# # 输出结果
# # 检查匹配距离是否小于一定阈值，以确保匹配的是正确的恒星
if d2d < 20 * u.arcsec:
    star = table[idx]
    # # 从Tycho-2星表中获取B和V波段的星等
    Bmag = star['BT']
    Vmag = star['VT']
    #
    # # 计算B-V的颜色指数
    b_v = Bmag - Vmag
    mag = Vmag + 0.09 * b_v
    #
    # # 输出结果
    print(f"In 20 Apparent magnitude: {mag:.2f}")
else:
    star = table[idx]

    Bmag = float(star['BT'])
    Vmag = float(star['VT'])
    #
    # # 计算B-V的颜色指数
    # b_v = Bmag - Vmag
    # mag = Vmag + 0.09 * b_v
    mag = Vmag - 2.5 * np.log10(data[y_pixel,x_pixel]) #+ zp

    # print('No star found within 20 arcsecond')
    print(f"Apparent magnitude: {mag:.2f}")


