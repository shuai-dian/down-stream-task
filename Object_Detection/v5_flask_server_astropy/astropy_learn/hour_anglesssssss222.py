from astropy.io import fits
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.wcs import WCS
# 读取FITS图像
image_file = fits.open(r"F:\ali_cloud\download\sat_00000.0185.fits")
image_data = image_file[0].data.astype(float)
header =  image_file[0].header
print("==========")
print(WCS(header))
print("==========")

print(type(header["CRPIX1"]))


header['SHUAI'] = "shuai"



# fits.writeto('example.fits', hdul[0].data, header, overwrite=True)
# print("==========")
# print(WCS(header))
# print("==========")




image_file[0].header["CRVAL1"] = 130 + 45 / 60 + 7 / 3600
image_file[0].header["CRVAL2"] = 46 +  31 / 60 + 12 / 3600
# 获取FITS图像的坐标系信息

for i in header:
    print(i,header[i])

wcs = WCS(image_file[0].header)
print(wcs)
# 确定卫星位置
satellite_position = (100, 200)  # 假设卫星位置为(100, 200)像素坐标
# 使用 WCS 将像素坐标转换为天球坐标
ra, dec = wcs.all_pix2world(100, 200, 0)
# 打印结果
print("方位: {:.2f} 度".format(ra))
print("赤纬: {:.2f} 度".format(dec))
aa = ra - 130.75
aa2 = dec - 46.52

# 确定观测位置和时间
location = EarthLocation.from_geodetic(lon=130.45 * u.deg, lat=46.31*u.deg, height=263*u.m)  # 上海市的地理坐标
print("location",location)
observation_time =Time('2023-02-22T20:01:40', format='isot', scale='utc')  # 观测时间为UTC时间
print("observation_time",observation_time)
# 计算卫星的赤道坐标
ra, dec = 0, 0  # 初始化赤经和赤纬
satellite_position_sky = SkyCoord.from_pixel(satellite_position[1], satellite_position[0], wcs=wcs, origin=0)
satellite_position_sky_fk5 = satellite_position_sky.transform_to('fk5')
ra, dec = satellite_position_sky_fk5.ra.deg, satellite_position_sky_fk5.dec.deg # # 130.75243856080868 46.520339998936194
print(ra,dec)

# 定义天体的位置
object_position = SkyCoord(ra=ra, dec=dec, unit='deg')
# 定义观察者的位置
#observer_location = EarthLocation(lat=35.20833333, lon=-111.635, height=2096)
# 计算天体的表观位置
apparent_object_position = object_position.transform_to(AltAz(obstime=observation_time, location=location))
print(apparent_object_position)
print(apparent_object_position.az)
print(apparent_object_position.alt)

# 计算天体的时角
hour_angle = observation_time.sidereal_time('apparent', longitude=location.lon) - apparent_object_position.az
print("hour_angle",hour_angle)

# 计算卫星的时角
#satellite_position_sky_altaz = satellite_position_sky_fk5.transform_to(AltAz(location=location, obstime=observation_time))
#hour_angle = satellite_position_sky_altaz.hourangle.deg
'''

# 计算卫星的地平坐标
azimuth = satellite_position_sky_altaz.az.deg
elevation = satellite_position_sky_altaz.alt.deg

# 可视化结果
fig, ax = plt.subplots()
ax.imshow(image_data, cmap='gray', origin='lower')

# 在图像中添加卫星位置的标记
ax.plot(satellite_position[1], satellite_position[0], 'rx')

# 在图像中添加赤道坐标和地平坐标信息
ax.text(0.05, 0.95, f'RA={ra:.2f} deg, Dec={dec:.2f} deg', transform=ax.transAxes, color='w', ha = 'left', va='top')
ax.text(0.05, 0.90, f'Az={azimuth:.2f} deg, El={elevation:.2f} deg', transform=ax.transAxes, color='w', ha='left', va='top')

plt.show()



import time
import  numpy as np
import math
import datetime
#观测地点的经度，#单位度
longitude = 130.0
time_str = "2023-02-22 20:01:40"
obs_time = datetime.datetime.strptime(time_str,"%Y-%m-%d %H:%M:%S")
j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
delta_t = (obs_time - j2000).total_seconds()
gmst = 18.697374558 + 24.06570982441908 * delta_t / 3600.0
gmst = gmst % 24
# 计算当地恒星时
lst = gmst + longitude / 15.0
lst = lst % 24
# 输出结果
print("当前时刻的地方恒星时为：", lst, "小时")
#卫星时角
A = 195.63863679
E = 36.0068921113
print("A:",A)
d_A = A * (180 / np.pi)
print("d_A:",d_A)

sat_hour_angle = lst - d_A

print(sat_hour_angle)
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
# 读取FITS图像
image_file = fits.open(r"F:\ali_cloud\download\sat_00000.0185.fits")
image_data = image_file[0].data.shape
print(image_data)
# 获取该像素点在图像中的位置
pixel_x = 512  # 假设该像素点在图像中的x坐标为100
pixel_y = 512  # 假设该像素点在图像中的y坐标为200
for i in image_file[0].header:
    print(i,image_file[0].header[i])

image_file[0].header["CRVAL1"] = 130 + 45 / 60 + 7 / 3600
image_file[0].header["CRVAL2"] = 46 +  31 / 60 + 12 / 3600
# 获取FITS图像的坐标系信息
wcs = WCS(image_file[0].header)
print(wcs)
# 计算该像素点在天球上的位置
coord = wcs.pixel_to_world(pixel_x, pixel_y)
print('======')
print(coord)
print('======')
# 获取赤经和赤纬信息
pixel_ra = coord.ra.value
pixel_dec = coord.dec.value
print(f"({pixel_x}, {pixel_y}) -> ({pixel_ra}, {pixel_dec})")
location = EarthLocation.from_geodetic(lon=130.45*u.deg, lat=46.31*u.deg, height=263*u.m)  # 上海市的地理坐标
print(location)
# 确定天球坐标系
# frame = 'fk5'  # 假设该FITS图像使用FK5坐标系3-02-22 20:01:40
frame = 'icrs'  # 假设该FITS图像使用FK5坐标系3-02-22 20:01:40
time = Time('2023-02-22T20:01:40')  # 假设观测时间为2023年3月5日中午12点
print(time)
# 计算本地子午线的时角
local_hour_angle = lst - pixel_ra
print("local_hour_angle",local_hour_angle)
# 计算该像素点在天球上的时角
hour_angle = local_hour_angle - pixel_ra/15
# 计算该像素点在本地天空中的位置
pixel_ra = coord.ra
pixel_dec = coord.dec

coord_2 = SkyCoord(pixel_ra, pixel_dec,frame=frame, equinox='J2000')
print("RA:", coord_2.ra)
print("Dec:", coord_2.dec)
print("Distance:", coord_2.distance)
print("Obstime:", coord_2.obstime)


altaz = coord.transform_to(AltAz(obstime=time, location=location))
# 输出结果
print('该像素点在本地天空中的位置：', altaz)
print('该像素点在天球上的时角：', hour_angle.to(u.hourangle))

'''