import astropy.io.fits as pyfits
import astropy
from astropy import nddata


ccd = astropy.nddata.CCDData.read("sat_00000.0101.fits")

print(ccd)
