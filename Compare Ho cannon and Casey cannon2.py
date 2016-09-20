'''Import the mathematica tools'''

###############################################################
# This time, do it without setting the scattering
######################################################
import matplotlib.pyplot as plt
from multiprocessing import Process, freeze_support
import cPickle
import csv
import numpy as np
from numpy import genfromtxt
from astropy.table import Table
import matplotlib.gridspec as gridspec



'''import the model---TheCannon 2'''
from astropy.table import Table
import AnniesLasso as tc



'''import the model --- TheCannon 1'''
import TheCannon
from TheCannon import apogee
from TheCannon import dataset
from TheCannon import model
from TheCannon import train_model

# read table and make the datasheet

tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("C:/Users/Jason/Downloads/example_DR10/example_DR10/Data")
tr_label = apogee.load_labels('C:/Users/Jason/Downloads/example_DR10/example_DR10/reference_labels.csv')
training_set_table = Table.read('C:/Users/Jason/Downloads/example_DR10/example_DR10/reference_labels.csv')

test_ID = tr_ID
test_flux = tr_flux
test_ivar = tr_ivar

ds = dataset.Dataset(wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)
ds.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

#############################################################################################
## Make some plot of the cannon 1

#fig = ds.diagnostics_SNR()
#mat.show(fig)

#fig_2 = ds.diagnostics_ref_labels()
#mat.show(fig_2)
###############################################################################################



##############################################################################################\
# obtain the normalized flux and arrange them
ds.ranges = [[371,3192], [3697,5500], [5500,5997], [6461,8255]]
pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=0.90, delta_lambda=50)
contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)

ds.set_continuum(contmask)
cont = ds.fit_continuum(3, "sinusoid")

norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = ds.continuum_normalize(cont)


ds.tr_flux = norm_tr_flux
ds.tr_ivar = norm_tr_ivar
ds.test_flux = norm_test_flux
ds.test_ivar = norm_test_ivar

# save file
with open("normalize flux.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(norm_tr_flux)

with open("normalize ivar.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(norm_tr_ivar)
#save SNR

###################################################################################################

# Train the model 1 and 2

md = model.CannonModel(2)
md.fit(ds)


md2 = tc.L1RegularizedCannonModel(training_set_table,norm_tr_flux,norm_tr_ivar)

#train the cannon2

# set the regularization strength
md2.regularization = 0


# set the label_vector and vectorized order

md2.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    training_set_table, tc.vectorizer.polynomial.terminator(
        ["Teff_{corr}", "logg_{corr}", "[M/H]_{corr}"], 2))
# print the information
print("Vectorizer terms: {0}".format(" + ".join(md2.vectorizer.get_human_readable_label_vector())))
# optimized the s2(let it be 0)
md2.s2 = 0


# Let's train the model
md2.train()

# solve the s2 value at each pixel. Make sure the chi-square is 0 and re-train
md2._set_s2_by_hogg_heuristic()
md2.train()

# check whether the model is trained
print md2.is_trained
print(md2)
# save model 2
md2.save("training-3 cannon2.pkl")
#data = cPickle.load(open("trainging-3 cannon2.pkl","rb"))
#output=open("training-3-h.txt","w")
#output.write(str(data))


###################################################################################################

# compare the plot of cannon 1 and 2
###########################################################################

# coefficients for cannon 1 and 2

thetac1 = md.coeffs
with open("theta cannon1.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(thetac1)


thetac2 = md2.theta
with open("theta cannon2.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(thetac2)




# inferred labels for the cannon 1 and 2

label_errs = md.infer_labels(ds)
infered_labelsc1 = ds.test_label_vals

with open("infered_labels cannon1 labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(infered_labelsc1)



infered_labelsc2 = md2.fit(norm_tr_flux, norm_tr_ivar)

# save the inferred-labels
with open("infered_labels cannon2 labels.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(infered_labelsc2)


# covariance matrix
# label_errs = md.infer_labels(ds)

# design matrix
design_matrix_c2 = md2.design_matrix
with open("design_matrix_c2.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(design_matrix_c2)


# compare plot
###############################
# plot chi-square
#md.diagnostics_plot_chisq(ds)
#tc.chi_sq(thetac2,design_matrix_c2,)


ds.diagnostics_test_step_flagstars()



# plot 1 to 1

# 1 to 1 for the cannon 1
snr = ds.test_SNR
label_names = ds.get_plotting_labels()
nlabels = len(label_names)
reference_labels = tr_label
test_labels = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/infered_labels cannon1 labels.csv', delimiter=',')

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

            figname_full = "%s_%s.png" %("cannon1", i)
            plt.savefig(figname_full)
            print("Diagnostic for label output vs. input")
            print("Saved fig %s" % figname_full)
            plt.close()


# 1 to 1 plot the cannon 2
snr = ds.test_SNR
label_names = ds.get_plotting_labels()
nlabels = len(label_names)
reference_labels = tr_label
test_labels = genfromtxt('C:/Users/Jason/Desktop/NYU/Laboratory/task 2016.8.1-12.23/My codes/2016.9-10 testing and plot/infered_labels cannon2 labels.csv', delimiter=',')

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




















