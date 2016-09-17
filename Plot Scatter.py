from numpy import genfromtxt
import AnniesLasso as tc
import csv
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib
import numpy

wl = norm_tr_flux = genfromtxt("wl")

scatter = numpy.load("scatter.npy")

print scatter
print scatter.shape
