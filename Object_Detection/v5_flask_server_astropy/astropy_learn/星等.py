from astropy import units as u
from astropy import constants as const
import math
from numpy import pi
from astropy.io import fits
from astropy import units as u


def get_star_level():

    hdulist = fits.open(r"F:\ali_cloud\download/sat_00000.0185.fits")
    data = hdulist[0].data

    # 假设卫星位于图像中心，使用一个15x15像素的框来测量卫星亮度
    box_size = 15
    x_center = data.shape[0] // 2
    y_center = data.shape[1] // 2
    box_data = data[x_center-box_size//2:x_center+box_size//2, y_center-box_size//2:y_center+box_size//2]
    peak_flux = box_data.max() * u.Jy

    print(peak_flux)
    # 维京1号星的亮度
    vega_flux = 3.52e-23 * u.W / u.m**2 / u.Hz
    # 计算卫星相对于参考星的亮度比，并使用log10函数计算星等
    m = -2.5 * math.log10((peak_flux / vega_flux).value)
    print("该天体的星等为：", m)
