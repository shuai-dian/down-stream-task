from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation

#open thr FITS file
hdulist = fits.open(r"F:\ali_cloud\download/sat_00000.0185.fits")
data = hdulist[0].data
wcs = WCS(hdulist[0].header)
for i in hdulist[0].header:
    print(i,hdulist[0].header[i])

print(wcs)
x_center = data.shape[1] / 2
y_center = data.shape[0] / 2

x, y = 100, 200 # 假设想要获取 (100, 200) 像素点的坐标
ra, dec = wcs.all_pix2world(x, y, 0, ra_dec_order=True)
print(ra, dec)
lon = 182.6357541667
lat = 39.405675
height = 55
observer = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)
obs_time = Time('2023-03-03 00:00:00') # 设定观测时间
sky_pos = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs') # 将赤道坐标转换为天体坐标
altaz = sky_pos.transform_to(AltAz(obstime=obs_time, location=observer))
hour_angle = altaz.obstime.sidereal_time('apparent', longitude=lon) - altaz.az

print(hour_angle)




# sky_coord = wcs.pixel_to_world(x_center,y_center)
# ra = sky_coord.ra.to(u.hourangle)
# dec = sky_coord.dec
# print(f"卫星位置的时角坐标: {ra}, 赤纬坐标: {dec}")


# x,y = np.meshgrid(np.arange(hdulist[0].header["NAXIS1"]),
#                   np.arange(hdulist[0].header["NAXIS2"]))

# convert the pixel coordinates  to sku coordinates

# print(x,y)
#
# ra,dec =  w.all_pix2world(x,y,0)
# print(ra,dec )
#
# hdulist.close()