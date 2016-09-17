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

for i in range(nlabels):
            name = label_names[i]
            orig = reference_labels[:,i]
            cannon = test_labels[:,i]
            # calculate bias and scatter
            scatter = np.round(np.std(orig-cannon),5)
            bias  = np.round(np.mean(orig-cannon),5)

            low = np.minimum(min(orig), min(cannon))
            high = np.maximum(max(orig), max(cannon))

            fig = plt.figure(figsize=(10,6))
            gs = gridspec.GridSpec(1,2,width_ratios=[2,1], wspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
            ax1.set_xlim(low, high)
            ax1.set_ylim(low, high)
            ax1.legend(fontsize=14, loc='lower right')
            pl = ax1.scatter(orig, cannon, marker='x', c=snr,
                    vmin=50, vmax=200, alpha=0.7)
            cb = plt.colorbar(pl, ax=ax1, orientation='horizontal')
            cb.set_label('SNR from Test Set', fontsize=12)
            textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                    fontsize=14, verticalalignment='top')
            ax1.tick_params(axis='x', labelsize=14)
            ax1.tick_params(axis='y', labelsize=14)
            ax1.set_xlabel("Reference Value", fontsize=14)
            ax1.set_ylabel("Cannon Test Value", fontsize=14)
            ax1.set_title("1-1 Plot of Label " + r"$%s$" % name)
            diff = cannon-orig
            npoints = len(diff)
            mu = np.mean(diff)
            sig = np.std(diff)
            ax2.hist(diff)
            #ax2.hist(diff, range=[-3*sig,3*sig], color='k', bins=np.sqrt(npoints),
            #        orientation='horizontal', alpha=0.3, histtype='stepfilled')
            ax2.tick_params(axis='x', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)
            ax2.set_xlabel("Count", fontsize=14)
            ax2.set_ylabel("Difference", fontsize=14)
            ax2.axhline(y=0, c='k', lw=3, label='Difference=0')
            ax2.set_title("Training Versus Test Labels for $%s$" %name,
                    fontsize=14)
            ax2.legend(fontsize=14)

            figname_full = "%s_%s.png" %("cannon2", i)
            plt.savefig(figname_full)
            print("Diagnostic for label output vs. input")
            print("Saved fig %s" % figname_full)
            plt.close()



"""
ds.diagnostics_test_step_flagstars()
ds.diagnostics_survey_labels()
ds.diagnostics_1to1()
"""
#save scattering


"""

"""
np.save("scatter",scatter)
np.savetxt("scatter.txt",scatter, delimiter=",")











