import numpy as np
from astropy.io import fits
import datetime

now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')

data = np.random.rand(2048, 2048)

hdu = fits.PrimaryHDU(data)
hdu.header["BITPIX"] = 16
hdu.header["NAXIS"] = 2
hdu.header["NAXIS1"] = 2048
hdu.header["NAXIS2"] = 2048
hdu.header["NAME"] = '107756'
hdu.header["DATE-OBS"] = now
hdu.header["HUMIDITY"]= 97.5
hdu.header["TEMP"] = 21.3
hdu.header["PRESSURE"] = 98.1
hdu.header["A"] = 203.122409999819
hdu.header["E"] = 31.6131159006528
hdu.header["MODE"] = 0

hdu.writeto('creat.fits', overwrite=True)

# image_hdu = fits.ImageHDU(n2)
# image_hdu2 = fits.ImageHDU(n3)
#
# c1 = fits.Column(name='a', array=np.array([1, 2]), format='K')
# c2 = fits.Column(name='b', array=np.array([4, 5]), format='K')
# c3 = fits.Column(name='c', array=np.array([7, 8]), format='K')
# table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3])
#
# hdul = fits.HDUList([primary_hdu, image_hdu, table_hdu])
#
#
# hdul.append(image_hdu2)
# hdr = fits.Header()
# hdr['OBSERVER'] = 'Edwin Hubble'
# hdr['COMMENT'] = "Here's some commentary about this FITS file."
# empty_primary = fits.PrimaryHDU(header=hdr)
