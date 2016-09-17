
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
plt.plot(wl, 1*coefficient[:,5],linewidth = 5.0)
#plt.plot(wl, 1*coefficient[:,1], wl, coefficient[:,2], wl, coefficient[:,3],linewidth = 5.0)
fig.suptitle('Coefficients for Teff, log g, [Fe/H]', fontsize=40)
plt.xlabel('Wave Length', fontsize=38)
plt.ylabel('Coefficients', fontsize=36)

axes = plt.gca()
axes.set_xlim([15660,15780])
#axes.set_xlim([16160,16280])

#axes.set_ylim([0.8, 1.2])

plt.show()






