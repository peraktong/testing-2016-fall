
from numpy import genfromtxt
import AnniesLasso as tc
import csv
from astropy.table import Table
import matplotlib.pyplot as plt


# read table
training_set_table = Table.read('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/reference_labels.csv')
# open file
norm_tr_flux = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/normalize flux.csv', delimiter=',')
norm_tr_ivar = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/normalize ivar.csv', delimiter=',')

md = tc.L1RegularizedCannonModel(training_set_table, norm_tr_flux, norm_tr_ivar)
md.load("C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/trainging-3.pkl")

wl = norm_tr_flux = genfromtxt("wl")
# save the file
coefficient = md.theta
with open("theta.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(coefficient)

# plot the Theta

plt.plot(wl,coefficient)
plt.show()
print coefficient
print len(coefficient)





