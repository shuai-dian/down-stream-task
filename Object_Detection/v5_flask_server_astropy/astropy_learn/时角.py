# from astropy import units as u
# from astropy.coordinates import SkyCoord, Distance
# from astropy.coordinates import FK5
# #度转时角
# c = SkyCoord(ra=4.1580*u.deg, dec=-5.7402*u.deg)
# print(c.to_string('hmsdms'))
# #时角转度
# coords = ["00:16:37.9 -05:44:25"]
# c = SkyCoord(coords, frame=FK5, unit=(u.hourangle, u.deg))
# print(c)
from astropy.wcs import WCS
from astropy.io import fits
# Open the FITS file and read the WCS information
hdulist = fits.open(r'F:\ali_cloud\download/sat_00000.0185.fits')
wcs = WCS(hdulist[0].header)
# Convert the pixel coordinates to celestial coordinates
x = 100
y = 200

ra, dec = wcs.all_pix2world(x, y, 0)
print(f'RA: {ra}, Dec: {dec}')
# from astropy.time import Time
# import astropy.units as u
# # 定义日期和时间
# date = '2022-12-20 12:00:00'
# # 将日期和时间转换为 astropy Time 对象
# t = Time(date)
# # 获取当前位置的地理经度（以角度为单位）
# longitude = 60 * u.degree
# # 计算时角坐标（以角度为单位）
# lst = t.sidereal_time('mean', longitude)
# # 打印时角坐标
# print(lst)




##时角坐标
'''
在astropy中，astropy.coordinates.EarthLocation.from_geodetic()函数用于定义地球上的一个位置。该函数需要三个参数：经度（lon）、纬度（lat）和高度（height）。
其中，height表示观测位置的高度，即相对于海平面的高度，通常以米为单位。在天文观测中，高度信息通常非常重要，因为它可以影响到观测条件、大气折射等因素，从而影响观测结果的精度。因此，在定义一个地点时，需要尽可能精确地输入高度信息。
这个height是相对于海平面的高度

'''
def get_time_angle():
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    import astropy.units as u

    # 输入观测时间和地点信息
    obs_time = Time('2022-03-01T22:30:00')  # 观测时间
    obs_location = EarthLocation.from_geodetic(lon='114.31d', lat='30.52d', height=40 * u.m)  # 观测地点

    # 输入像素坐标信息
    x_pix = 100  # 像素x坐标
    y_pix = 200  # 像素y坐标

    # 将像素坐标转换为天球坐标
    image_coord = SkyCoord(x_pix, y_pix, frame='pixels')

    # 将天球坐标转换为时角坐标
    altaz = image_coord.transform_to(AltAz(obstime=obs_time, location=obs_location))
    print("时角坐标: ", altaz.ra)



'''可以使用Tycho-2星表和Astropy软件包来估算未知人造卫星的星等。具体步骤如下：
从Tycho-2星表中获取一些亮星的位置和星等信息。
使用Astropy中的SkyCoord函数将这些亮星的位置转换为赤道坐标。
观测目标人造卫星，并测量它在图像中的像素位置。
使用Astropy中的WCS函数将像素位置转换为赤道坐标。
根据卫星在赤道坐标系中的位置，计算其与每个亮星的角距离和角距离对应的星等。
使用加权平均法估算未知卫星的星等。
'''
def get_star_level():
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy import units as u
    import numpy as np
    # 读取Tycho-2星表文件
    data = np.loadtxt('tyc2.dat', usecols=(1, 2, 3, 4, 5, 6))
    ra = data[:, 0]
    dec = data[:, 1]
    mag = data[:, 3]


    # 将亮星的位置转换为赤道坐标
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    # 读取观测图像并获取WCS对象
    hdul = fits.open('obs_image.fits')
    wcs = WCS(hdul[0].header)
    # 测量目标卫星在图像中的像素位置
    x_pixel = 500
    y_pixel = 600
    # 将像素位置转换为赤道坐标
    c_pixel = wcs.pixel_to_world(x_pixel, y_pixel)
    # 计算目标卫星与每个亮星的角距离和对应的星等
    distances = c.separation(c_pixel)
    mag_distances = mag + 5 * np.log10(distances.to(u.pc).value / 10)
    # 使用加权平均法估算未知卫星的星等
    weights = 1 / distances.value
    unknown_mag = np.average(mag_distances, weights=weights)
    print("未知卫星的星等为：", unknown_mag)


def star_level2():
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.io import ascii
    from astropy.table import Table
    from astropy import coordinates as coord
    # 加载tycho-2星表数据
    tycho2_data = Table.read('tycho2.fits')
    # 观测时间
    obs_time = Time('2023-03-03T00:00:00', format='isot', scale='utc')
    # 观测位置
    obs_location = EarthLocation(lat=30.52 * u.deg, lon=114.31 * u.deg, height=40 * u.m)
    # 未知人造卫星的位置（假设已知）
    satellite_coord = SkyCoord(ra=10 * u.deg, dec=20 * u.deg, distance=100 * u.km, frame='icrs')
    # 将人造卫星的位置转换为赤道坐标系下的位置
    satellite_eq_coord = satellite_coord.transform_to(coord.ICRS)
    # 计算人造卫星到观测者的距离
    satellite_distance = obs_location.geocentric_distance - satellite_eq_coord.distance
    # 计算人造卫星的视星等
    satellite_mag = tycho2_data['BTmag'][0] + 5 * math.log10(satellite_distance.value / 10)
    print("未知人造卫星的视星等为：", satellite_mag)

def get_star_level3():
    import math
    # 卫星或者太空碎片的绝对星等
    M = -5.0
    # 卫星或者太空碎片与观测者之间的距离（单位：千米）
    distance = 1000
    # 观测地点的经度和纬度（单位：度）
    latitude = 40
    longitude = -90
    # 观测时间（UTC时间）
    observation_time = '2022-03-03T12:00:00Z'
    # 大气校正的修正量
    airmass = 1.0 / math.sin(math.radians(90 - latitude))\
    # 计算视星等
    V = M + 5 * math.log10(distance) - 5 - 0.2 * airmass
    print("视星等为：", V)


def get_star_level4():
    import math

    # 输入目标星体的赤经、赤纬，以及Tycho-2星表文件路径，返回目标星体的视星等和视差
    def get_target_star_info(ra, dec, tycho2_file):
        with open(tycho2_file, 'r') as f:
            for line in f:
                # 每行的格式为:
                # TYC1 TYC2 HIP RA(J2000) Dec(J2000) Vmag pmRA pmDec e_RA e_Dec e_Vmag Nc Flag
                fields = line.split()
                if ra == float(fields[3]) and dec == float(fields[4]):
                    V = float(fields[5])  # 视星等
                    parallax = float(fields[8])  # 视差，单位为毫弧秒
                    return (V, parallax)
        # 如果在Tycho-2星表中找不到目标星体，返回None
        return None

    # 输入目标星体的视星等和视差，计算目标星体的星等
    def calculate_magnitude(V, parallax):
        d = 1000 / parallax  # 将视差从毫弧秒转换为秒差距
        m = V + 5 * math.log10(d) - 5
        return m

    # 示例代码
    tycho2_file = 'tycho2.dat'  # Tycho-2星表文件路径
    ra = 0.0  # 目标星体的赤经，单位为度
    dec = 0.0  #
    target_info = get_target_star_info(ra, dec, tycho2_file)
    if target_info is not None:
        V, parallax = target_info
        # 计算目标星体的星等
        magnitude = calculate_magnitude(V, parallax)
        print(f"The magnitude of the target star is: {magnitude:.2f}")
    else:
        print("The target star cannot be found in the Tycho-2 catalog.")


def get_star_level5():
    import math

    # 输入星星的视星等和视差，计算星等
    def calculate_magnitude(V, parallax):
        d = 1000 / parallax  # 将视差从毫弧秒转换为秒差距
        m = V + 5 * math.log10(d) - 5
        return m

    # 读取Tycho-2星表文件，获取星星的J2000坐标和视星等
    def read_tycho2_file(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                # 每行的格式为:
                # TYC1 TYC2 HIP RA(J2000) Dec(J2000) Vmag pmRA pmDec e_RA e_Dec e_Vmag Nc Flag
                fields = line.split()
                ra = float(fields[3])  # J2000赤经，单位为度
                dec = float(fields[4])  # J2000赤纬，单位为度
                V = float(fields[5])  # 视星等
                parallax = float(fields[8])  # 视差，单位为毫弧秒
                data.append((ra, dec, V, parallax))
        return data

    # 示例代码
    tycho2_file = 'tycho2.dat'  # Tycho-2星表文件路径
    data = read_tycho2_file(tycho2_file)
    ra, dec, V, parallax = data[0]  # 取第一个星星的数据
    m = calculate_magnitude(V, parallax)  # 计算星等
    print('RA: {:.2f}, Dec: {:.2f}, V: {:.2f}, Parallax: {:.2f}, Magnitude: {:.2f}'.format(ra, dec, V, parallax, m))