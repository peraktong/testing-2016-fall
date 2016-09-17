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
training_set_table = Table.read('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/reference_labels.csv')
# open file
norm_tr_flux = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/normalize flux.csv', delimiter=',')
norm_tr_ivar = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/normalize ivar.csv', delimiter=',')


# load the training of the cannon2 model
md = tc.L1RegularizedCannonModel(training_set_table, norm_tr_flux, norm_tr_ivar)
md.load("C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/trainging-3.pkl")
print md.is_trained


#####################################################################
#  test the model:
#####################################################################


infered_labels = md.fit(norm_tr_flux, norm_tr_flux)

"""
# save the inferred-labels
infered_labels = md.fit(a,b)

with open("infered_labels-3 labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(infered_labels)



# print the inferred labels:
print infered_labels
########################
"""


#########################################################
# Let's try the DR10 data and try to infer the labels from the fits file of spectrum
################################################################################


infered_labels = md.fit(norm_tr_flux[:],norm_tr_ivar[:])
with open("infered_labels-3 labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(infered_labels)


md.theta
