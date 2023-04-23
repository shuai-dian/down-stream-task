from detector import Detector
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import math
from astropy import units as u
from astropy import constants as const
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from numpy import pi

detector = Detector()

def process_fits(image_data):
    n, bins, patches = plt.hist(image_data.flatten(), 2560)
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
    return img_data
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
table = Table.read('tycho2.fits', format='fits')
coords = SkyCoord(ra=table['RAdeg'], dec=table['DEdeg'], unit=(u.hourangle, u.deg), frame='icrs')
print(coords)
mags = table['VT']
hdulist = fits.open(r"F:\ali_cloud\download\sat_00000.0185.fits")
header = hdulist[0].header
data = hdulist[0].data
img = process_fits(data)
cv2.imwrite("01.png",img)
img = cv2.imread("01.png")
boxes = detector.detect(img)
print(boxes)
wcs = WCS(hdulist[0].header)
print(wcs)

for i in boxes:
    print(i)
    x1,y1,x2,y2 = i[0],i[1],i[2],i[3]
    print()
    x_pixel = x1 + (x2 - x1) // 2
    y_pixel = y1 + (y2 - y1) // 2
    box_data = data[x1:x2,y1:y2].max() * u.Jy
    print("box_data",box_data)

    # 将像素坐标转换为天球坐标
    ra, dec = wcs.all_pix2world(x_pixel, y_pixel, 0)
    print(ra,dec)
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
        # # 计算B-V的颜色指数
        # b_v = Bmag - Vmag
        # mag = Vmag + 0.09 * b_v
        mag = Vmag - 2.5 * np.log10(data[y_pixel, x_pixel])  # + zp
        # print('No star found within 20 arcsecond')
        print(f"Apparent magnitude: {mag:.2f}")


# 星等获取方法1
def method1(data,boxes):
    # 假设卫星位于图像中心，使用一个15x15像素的框来测量卫星亮度
    box1 = boxes[0]
    box_data = data[box1[0]:box1[2], box1[1]:box1[3]]
    peak_flux = box_data.max() * u.Jy
    print(peak_flux)
    # 维京1号星的亮度
    vega_flux = 3.52e-23 * u.W / u.m**2 / u.Hz
    # 计算卫星相对于参考星的亮度比，并使用log10函数计算星等
    m = -2.5 * math.log10((peak_flux / vega_flux).value)
    print("method1 该天体的星等为：", m)
#method1(data,boxes)

def method2(header):
    # 获取WCS对象
    wcs = WCS(header)
    # 获取已知星体的坐标和星等
    known_ra = 10.0  # 已知星体的赤经（度）
    known_dec = 20.0  # 已知星体的赤纬（度）
    known_mag = 15.0  #
    # 将已知星体的坐标转换为天球坐标
    known_coord = SkyCoord(known_ra, known_dec, unit='deg')
    # 将已知星体的坐标转换为像素坐标
    known_pix = wcs.world_to_pixel(known_coord)
    print(int(known_pix[0]),int(known_pix[1]))
    # 获取图像数据
    # 计算像素之间的距离（以度为单位）
    cdelt = np.abs(header['CDELT1'])  # 每个像素的角度大小
    dist = np.sqrt((np.arange(data.shape[0]) - known_pix[1]) ** 2 +
                   (np.arange(data.shape[1]) - known_pix[0]) ** 2) * cdelt
    # 计算其他像素的星等
    other_mags = known_mag + 2.5 * np.log10(data / np.min(data)) + 5 * np.log10(dist)
    print(other_mags)
#method2(header)

def method3(data,header):
    # 获取WCS对象
    wcs = WCS(header)
    # 获取已知星体的坐标和星等
    known_ra = 10.0  # 已知星体的赤经（度）
    known_dec = 20.0  # 已知星体的赤纬（度）
    known_mag = 15.0  # 已知星体的星等
    # 将已知星体的坐标转换为天球坐标
    known_coord = SkyCoord(known_ra, known_dec, unit='deg')
    # 将已知星体的坐标转换为像素坐标
    known_pix = wcs.world_to_pixel(known_coord)
    # 计算像素之间的距离（以度为单位）
    cdelt = np.abs(header['CDELT1'])  # 每个像素的角度大小
    dist = np.sqrt((np.arange(data.shape[0]) - known_pix[1]) ** 2 +
                   (np.arange(data.shape[1]) - known_pix[0]) ** 2) * cdelt
    # 计算其他像素的星等
    other_mags = known_mag + 2.5 * np.log10(data / np.min(data)) + 5 * np.log10(dist)
    print(other_mags)
#method3(data,header)


def method4(header):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # 已知的像素坐标和星等
    ra_known, dec_known = 10.0, 20.0
    mag_known = 15.0
    # 将像素坐标转换为天球坐标
    skycoord_known = SkyCoord(ra_known * u.deg, dec_known * u.deg, frame='icrs')
    # 获取头文件中的望远镜位置和观测时间信息
    telescope_location = ("2650", ('130d45m07s','46d31m12s', 263))
    observation_time = header['DATE-OBS']
    # 计算目标像素在天球上的位置
    skycoord_target = skycoord_known.apply_space_motion(time=observation_time, location=telescope_location)

    # 从tycho-2星表中读取星等信息
    from astroquery.vizier import Vizier
    from astropy.coordinates import match_coordinates_sky

    # 定义目标位置的搜索半径
    search_radius = 5.0 * u.arcsec

    # 在tycho-2星表中搜索距离目标位置最近的恒星
    catalogs = Vizier(columns=['TYC', 'VTmag'], row_limit=1).query_region(skycoord_target, radius=search_radius)
    if catalogs:
        catalog = catalogs[0]
        matched_star_coord = SkyCoord(ra=catalog['RAJ2000'][0], dec=catalog['DEJ2000'][0], unit=u.deg)
        matched_star_mag = catalog['VTmag'][0]

        # 计算目标像素的星等
        distance = skycoord_target.separation(matched_star_coord)
        mag_target = matched_star_mag + 5 * (np.log10(distance.to(u.pc).value) - 1)

        print(f"The estimated magnitude of the target pixel is {mag_target:.2f}.")
    else:
        print("No matching star is found in the catalog.")
#for i in header:
#    print(i,header[i])
#method4(header)

#
def tex_loc(header):
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    Teltk_ra = header["TELTKRA"]
    TELTKDEC = header["TELTKDEC"]
    # 定义观测地点的地理坐标
    observing_location = EarthLocation(lat='130d45m07s', lon='46d31m12s', height=263)
    observing_time = header["DATE-OBS"]
    print(observing_time)
    # 定义观测时间
    observing_time = Time.now()
    # 定义一个时角坐标
    ha = '02h30m00s'
    dec = '+60d00m00s'
    coord = SkyCoord(ha, dec, frame='icrs')
    # 将时角坐标转换为赤经坐标
    coord = coord.transform_to('fk5')
    # 计算当前天顶角
    altaz = coord.transform_to(AltAz(obstime=observing_time, location=observing_location))
    zenith_angle = altaz.alt.deg
    print('赤经坐标:', coord.ra)
    print('当前天顶角:', zenith_angle)

#tex_loc(header)
#天球坐标系，转时角坐标系
def sky_2_altaz():
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    # 定义观测地点的地理坐标
    observing_location = EarthLocation(lat='40d43m30s', lon='-74d00m23s', height=10)
    # 定义观测时间
    observing_time = Time.now()
    # 从名称字符串中创建天球坐标
    coord = SkyCoord.from_name('M31')
    # 将天球坐标转换为时角坐标
    ha = coord.transform_to(AltAz(obstime=observing_time, location=observing_location)).az
    print('时角坐标:', ha)
#sky_2_altaz()



#
# #header中主要目标的赤经和赤纬
# RA = header["CENTAZ"]
# L = header["CENTALT"]



