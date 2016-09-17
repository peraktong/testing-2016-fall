# import the data
from TheCannon import apogee
from TheCannon import dataset
import csv
from TheCannon import model
import AnniesLasso as tc
import numpy as np
import matplotlib.pyplot as plt
from helpers.corner import corner
from matplotlib.ticker import MaxNLocator
from copy import deepcopy

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


ds.tr_flux = norm_tr_flux
ds.tr_ivar = norm_tr_ivar
ds.test_flux = norm_test_flux
ds.test_ivar = norm_test_ivar

md = model.CannonModel(2)
md.fit(ds)


#md2 = tc.L1RegularizedCannonModel(tr_label, norm_tr_flux, norm_tr_ivar)
#md2.load("C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/trainging-3.pkl")

# use the function
label_names = ds.get_plotting_labels()
lams = ds.wl
pivots = md.pivots
npixels = len(lams)
nlabels = len(pivots)
chisqs = md.chisqs
coeffs = md.theta
first_order = coeffs[:,1:1+nlabels]
scatters = md.scatters

figname = "theta v2 try"
# triangle plot of the higher-order coefficients
labels = [r"$%s$" % l for l in label_names]
fig = corner(first_order, labels=labels, show_titles=True,
                     title_args = {"fontsize":12})
filename = "leading_coeffs_triangle.png"
print("Diagnostic plot: triangle plot of leading coefficients")
fig.savefig(figname)
print("Saved as %s" %figname)
plt.close(fig)