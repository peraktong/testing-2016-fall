'''Import the mathematica tools'''
import matplotlib.pyplot as plt
from multiprocessing import Process, freeze_support
import cPickle
import csv
import numpy as np


'''import the model---TheCannon 2'''
from astropy.table import Table
import AnniesLasso as tc



'''import the model --- TheCannon 1'''
import TheCannon
from TheCannon import apogee
from TheCannon import dataset


############################################################################

# reader the table of labels and the spectrum

training_set_table = Table.read('C:/Users/Jason/Downloads/example_DR10/example_DR10/reference_labels.csv')
tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("C:/Users/Jason/Downloads/example_DR10/example_DR10/Data")

# package the data
test_ID = tr_ID
test_flux = tr_flux
test_ivar = tr_ivar

# set the list
ds = dataset.Dataset(wl, tr_ID, tr_flux, tr_ivar, training_set_table, test_ID, test_flux, test_ivar)
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

# Choose some spectrum range and make it continuous
ds.ranges = [[371,3192], [3697,5500], [5500,5997], [6461,8255]]
pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=0.90, delta_lambda=50)

#continuous mask
contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)
ds.set_continuum(contmask)

# fit
cont = ds.fit_continuum(3, "sinusoid")

# obtain the norm_flux norm_inverse variance and ...
norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)

# rename
normalized_flux = norm_tr_flux
normalized_ivar = norm_tr_ivar

#protect recursive
freeze_support()

# training of the cannon2 model
md = tc.L1RegularizedCannonModel(training_set_table, normalized_flux, normalized_ivar)

"""
May use s model by using all the cores
model2 = tc.CannonModel(training_set_table, normalized_flux, normalized_ivar,
    dispersion=dispersion, threads=-1)
"""
# set the regularization strength
md.regularization = 0


# set the label_vector and vectorized order

md.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    training_set_table, tc.vectorizer.polynomial.terminator(
        ["Teff_{corr}", "logg_{corr}", "[M/H]_{corr}"], 2))
# print the information
print("Vectorizer terms: {0}".format(" + ".join(md.vectorizer.get_human_readable_label_vector())))
# optimized the s2(let it be 0)
md.s2 = 0

# Let's train the model
md.train()

# solve the s2 value at each pixel. Make sure the chi-square is 0 and re-train
md._set_s2_by_hogg_heuristic()
md.train()

# check whether the model is trained
print md.is_trained
print(md)



# save model and make a human-readable version
"""
md.save("trainging-3.pkl")
data = cPickle.load(open("C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 training step/trainging-3.pkl","rb"))
output=open("training-3-h.txt","w")
output.write(str(data))

"""


#####################################################################
#  test the model:
#####################################################################


tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("M92")
inferred_labels = md.fit(normalized_flux, normalized_ivar)

# save the inferred-labels
with open("infered_labels-M92 labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(inferred_labels)


# plot the difference of the inferred label and the validation label
for i, label_name in enumerate(md.vectorizer.label_names):
    fig, ax = plt.subplots()

    x = training_set_table[label_name]

    y = inferred_labels[:, i]

    abs_diff = np.abs(y - x)

    ax.scatter(x, y, facecolor="k")

    limits = np.array([ax.get_xlim(), ax.get_ylim()])

    ax.set_xlim(limits.min(), limits.max())

    ax.set_ylim(limits.min(), limits.max())

    ax.set_title("{0}: {1:.2f}".format(label_name, np.mean(abs_diff)))

    print("{0}: {1:.2f}".format(label_name, np.mean(abs_diff)))

plt.show()

# print the inferred labels:
print inferred_labels
#################################################################################
# Let's try the DR10 data and try to infer the labels from the fits file of spectrum
################################################################################








