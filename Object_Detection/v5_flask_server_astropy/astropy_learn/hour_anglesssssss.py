import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_body, Angle
import time
import  numpy as np
from astropy.time import Time
import math
import datetime
#观测地点的经度，#单位度
longitude = 130.0
now = datetime.datetime.utcnow()
print(now)
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
print("A:",A,"E",E)
az = A * u.deg  # 方位角
alt = E * u.deg  # 俯仰角
loc = EarthLocation(lat=46.31*u.deg, lon=130.45*u.deg, height=263*u.m)  # 观测地点
# 观测时间
obs_time = Time('2023-02-22 20:01:40', scale='utc')  # UTC时间
# 将地平坐标转换为赤道坐标
coord = SkyCoord(alt=alt, az=az, frame='altaz', location=loc, obstime=obs_time)
equatorial_coord = coord.transform_to('icrs')
# 计算LST
lst = obs_time.sidereal_time('apparent', longitude=loc.lon) # 14h53m19.63281977s
# 计算时角
ha = Angle(lst - equatorial_coord.ra)
print('时角坐标:', ha.to_string(unit=u.hourangle)) # 0h51m55.73202799s
# 计算FOV
fov = 2 * coord.alt.to_value() * np.tan(0.5 * np.pi / 180)
print("fov",fov)
# 假设你已经知道了图像大小
image_size = 2048
# 计算像素大小
pixel_size = fov / image_size
print("pixel_size",pixel_size)
# 计算DELT和CD1_1
DELT = pixel_size * u.arcsec / u.pix
CD1_1 = -DELT
print("DELT",DELT,"CD1_1",CD1_1)





from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.time import Time
import astropy.units as u
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