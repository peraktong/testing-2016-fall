# import the data
from TheCannon import apogee
from TheCannon import dataset
import csv
from TheCannon import model
from numpy import genfromtxt
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

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



########################################################
#plot
snr = ds.test_SNR
label_names = ds.get_plotting_labels()
nlabels = len(label_names)
reference_labels = tr_label
test_labels = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 training step/infered_labels-3 labels.csv', delimiter=',')

i=0
orig = reference_labels[:,i]
cannon = test_labels[:,i]
# calculate bias and scatter
scatter = np.round(np.std(orig-cannon),5)
bias  = np.round(np.mean(orig-cannon),5)

np.save("scatter",scatter)











