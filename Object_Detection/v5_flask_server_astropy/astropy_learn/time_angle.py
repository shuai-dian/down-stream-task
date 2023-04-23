

'''import math
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
# 望远镜位置
observing_location = EarthLocation.from_geodetic(lon=120.0, lat=30.0, height=0*u.m)
# 望远镜当前时角
current_ha = 2.5 * u.hourangle
# 卫星在图像中的像素坐标
sat_pixel = [512, 512]
# 图像尺寸
image_size = [1024, 1024]
# 将像素坐标转换为角度坐标
fov = 1 * u.degree
center_pixel = [image_size[0]/2, image_size[1]/2]
x = (sat_pixel[0] - center_pixel[0]) / image_size[0] * fov
y = (sat_pixel[1] - center_pixel[1]) / image_size[1] * fov
angle_coord = SkyCoord(x, y, frame='icrs', unit='deg')

# 计算卫星在当前时刻的时角坐标
observing_time = Time.now()
observing_time = observing_time - current_ha
altaz_frame = AltAz(location=observing_location, obstime=observing_time)
icrs_coord = angle_coord.transform_to(ICRS())
altaz_coord = icrs_coord.transform_to(altaz_frame)
sat_ha = altaz_coord.az
sat_dec = altaz_coord.alt

# 输出卫星的时角坐标
print("卫星的时角坐标为：", sat_ha.to_string(unit=u.hourangle))
print("卫星的赤纬坐标为：", sat_dec.to_string())
'''


'''
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, get_body

# 设置观测位置
observing_location = EarthLocation.from_geodetic(lon=120.0, lat=30.0, height=0*u.m)

# 设置观测时间
observing_time = Time('2023-03-05T00:00:00')

# 计算卫星在给定时刻的天球坐标
satellite_name = 'ISS'  # 卫星名称
satellite_body = get_body(satellite_name, observing_time, observing_location)
satellite_ra, satellite_dec, satellite_distance = satellite_body.ra, satellite_body.dec, satellite_body.distance

# 输出卫星的天球坐标
print('卫星在观测位置的天球坐标：RA={}, DEC={}'.format(satellite_ra.to_string(unit=u.hourangle, sep=':'), satellite_dec.to_string(sep=':')))
'''
'''
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
import astropy.coordinates as coord

# 定义观测位置和观测时间
observing_location = EarthLocation.from_geodetic(lon=120.0, lat=30.0, height=0*u.m)
observing_time = Time('2023-03-05T00:00:00')

# 假设卫星在图像上的像素坐标为 (x, y)
x = 100
y = 200

# 将像素坐标转换为天球坐标
wcs = ...  # 使用您的方法获得图像的 WCS 信息
ra, dec = wcs.pixel_to_world(x, y)

# 创建 SkyCoord 对象
satellite_coords = SkyCoord(ra, dec, unit=u.deg)

# 将天球坐标转换为地平坐标
alt_az_frame = AltAz(location=observing_location, obstime=observing_time)
alt_az_coords = satellite_coords.transform_to(alt_az_frame)

# 输出卫星在观测位置的地平坐标
print('卫星在观测位置的地平坐标：ALT={}, AZ={}'.format(alt_az_coords.alt.to_string(unit=u.deg), alt_az_coords.az.to_string(unit=u.deg)))

'''
