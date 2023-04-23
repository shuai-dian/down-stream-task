import matplotlib.pyplot as plt
import sys
import astropy.io.fits as pyfits
import pywcsgrid2
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
imagename=str("data/20201220013535730_45696.fit")
# def setup_axes():
#     ax = pywcsgrid2.subplot(111, header=f_radio[0].header)
#     return ax

# GET IMAGE1
data = f_radio[0].data #*1000
ax = setup_axes()

# prepare figure & axes
fig = plt.figure(1)

#GET CONTOUR SOURCE:
f_contour_image = pyfits.open("IMAGE2.fits")
data_contour = f_contour_image[0].data

# DRAW CONTOUR
cont = ax.contour(data_contour, [5, 6, 7, 8, 9],
                   colors=["r","r","r", "r", "r"], alpha=0.5)

# DRAW IMAGE
im = ax.imshow(data, cmap=plt.cm.gray, origin="lower", interpolation="nearest",alpha=1.0)
plt.show()