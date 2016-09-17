# import the data
from TheCannon import apogee
from TheCannon import dataset
import csv
from TheCannon import model

import matplotlib.pyplot as mat



tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("C:/Users/Jason/Downloads/example_DR10/example_DR10/Data")
tr_label = apogee.load_labels('C:/Users/Jason/Downloads/example_DR10/example_DR10/reference_labels.csv')

with open("tr_flux.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(tr_flux)

with open("tr_ivar.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(tr_ivar)