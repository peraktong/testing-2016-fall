# import the data
from TheCannon import apogee
from TheCannon import dataset
import csv
from numpy import genfromtxt
from TheCannon import model

import matplotlib.pyplot as mat



tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("C:/Users/Jason/Downloads/example_DR10/example_DR10/Data")
tr_label = apogee.load_labels('C:/Users/Jason/Downloads/example_DR10/example_DR10/reference_labels.csv')

# package the data
test_ID = tr_ID
test_flux = tr_flux
test_ivar = tr_ivar

# import from the cannon

ds = dataset.Dataset(wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

# diagnose
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

#fig = ds.diagnostics_SNR()
#mat.show(fig)

#fig_2 = ds.diagnostics_ref_labels()
#mat.show(fig_2)

ds.ranges = [[371,3192], [3697,5500], [5500,5997], [6461,8255]]
pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=0.90, delta_lambda=50)
contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)

ds.set_continuum(contmask)
cont = ds.fit_continuum(3, "sinusoid")

norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)

# save file
with open("normalize flux.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(norm_tr_flux)

with open("normalize ivar.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(norm_tr_ivar)

# open file
norm_tr_flux = genfromtxt('normalize flux.csv', delimiter=',')
norm_tr_ivar = genfromtxt('normalize ivar.csv', delimiter=',')

