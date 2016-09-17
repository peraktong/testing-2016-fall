
from numpy import genfromtxt
import AnniesLasso as tc
import csv
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib


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
font = {'weight': 'bold',
            'size': 30}

matplotlib.rc('font', **font)
fig = plt.figure()
plt.plot(wl, coefficient[:,[0,2]])
fig.suptitle('Coefficients for Teff, log g, [Fe/H]', fontsize=40)
plt.xlabel('Wave Length', fontsize=38)
plt.ylabel('Coefficients', fontsize=36)

plt.show()
print coefficient
print len(coefficient)





