import astropy
from astropy.table import Table
from astropy.io import fits
hdul = fits.open("../sat_00000.0101.fits")
header = hdul[0].header
# data = hdul[0].data
# print(data)
# print(data.shape)
# print(data.dtype.name) # uint16




# print(hdul[0].header)
#
for i in hdul[0].header:
    print(i,header[i])