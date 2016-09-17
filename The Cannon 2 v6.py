'''Import the mathematica tools'''
import matplotlib.pyplot as plt
from multiprocessing import Process, freeze_support
import cPickle
import csv
import numpy as np
from numpy import genfromtxt



'''import the model---TheCannon 2'''
from astropy.table import Table
import AnniesLasso as tc



'''import the model --- TheCannon 1'''
import TheCannon
from TheCannon import apogee
from TheCannon import dataset

# read table
training_set_table = Table.read('C:/Users/Jason/Downloads/example_DR10/example_DR10/reference_labels.csv')
# open file
norm_tr_flux = genfromtxt('normalize flux.csv', delimiter=',')
norm_tr_ivar = genfromtxt('normalize ivar.csv', delimiter=',')


# training of the cannon2 model
md = tc.L1RegularizedCannonModel(training_set_table, norm_tr_flux, norm_tr_ivar)

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

md.save("trainging-3.pkl")
data = cPickle.load(open("trainging-3.pkl","rb"))
output=open("training-3-h.txt","w")
output.write(str(data))

# load a model
"""
md_2 = tc.L1RegularizedCannonModel(training_set_table, norm_tr_flux, norm_tr_ivar)
md_2.load("trainging-3.pkl")

"""



#####################################################################
#  test the model:
#####################################################################
"""
infered_labels = md.fit(norm_tr_flux, norm_tr_ivar)

# save the inferred-labels
with open("infered_labels-3 labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(infered_labels)



# print the inferred labels:
print infered_labels

"""

#################################################################################
# Let's try the DR10 data and try to infer the labels from the fits file of spectrum
################################################################################
