from numpy import genfromtxt
import AnniesLasso as tc
import csv
from astropy.table import Table
from AnniesLasso import diagnostics
import matplotlib.pyplot as plt
import numpy



coefficient = numpy.fromfile("theta.csv")
print len(coefficient)