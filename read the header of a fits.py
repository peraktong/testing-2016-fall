from astropy.io import fits
import matplotlib.pyplot as mat
hdulist = fits.open("apVisitSum-5608-55960.fits")
prihdr = hdulist[0].header
print(prihdr)
print len(prihdr)
print type(hdulist)
hdulist.info()
hdulist.close()

