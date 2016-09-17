from numpy import genfromtxt
import matplotlib.pyplot as plt
import random
import matplotlib

norm_tr_flux = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/normalize flux.csv', delimiter=',')

# plot normalized flux for 10 random star
wl = genfromtxt("wl")

p = 118
font = {'weight': 'bold',
            'size': 30}

matplotlib.rc('font', **font)
fig = plt.figure()
plt.plot(wl, norm_tr_flux[p, :],linewidth =5.0)
fig.suptitle(' Continuum-normalized spectra', fontsize=40)
plt.xlabel('Wave Length', fontsize=38)
plt.ylabel('Normalized flux', fontsize=36)
axes = plt.gca()
axes.set_xlim([15660,15780])
#axes.set_xlim([16160,16280])
axes.set_ylim([0.8,1.1])
plt.show()

