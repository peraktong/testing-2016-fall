import matplotlib as plt








# contmask
ds.ranges = [[371,3192], [3697,5500], [5500,5997], [6461,8255]]
pseudo_tr_flux, pseudo_tr_ivar = ds.continuum_normalize_training_q(q=0.90, delta_lambda=50)
contmask = ds.make_contmask(pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)




# define the plot_continous pixel
def plot_contpix(self, x, y, contpix_x, contpix_y, figname):
    """ Plot baseline spec with continuum pix overlaid

    Parameters
    ----------
    """
    fig, axarr = plt.subplots(2, sharex=True)
    plt.xlabel(r"Wavelength $\lambda (\AA)$")
    plt.xlim(min(x), max(x))
    ax = axarr[0]
    ax.step(x, y, where='mid', c='k', linewidth=0.3,
            label=r'$\theta_0$' + "= the leading fit coefficient")
    ax.scatter(contpix_x, contpix_y, s=1, color='r',
               label="continuum pixels")
    ax.legend(loc='lower right',
              prop={'family' :'serif', 'size' :'small'})
    ax.set_title("Baseline Spectrum with Continuum Pixels")
    ax.set_ylabel(r'$\theta_0$')
    ax = axarr[1]
    ax.step(x, y, where='mid', c='k', linewidth=0.3,
            label=r'$\theta_0$' + "= the leading fit coefficient")
    ax.scatter(contpix_x, contpix_y, s=1, color='r',
               label="continuum pixels")
    ax.set_title("Baseline Spectrum with Continuum Pixels, Zoomed")
    ax.legend(loc='upper right', prop={'family' :'serif',
                                       'size' :'small'})
    ax.set_ylabel(r'$\theta_0$')
    ax.set_ylim(0.95, 1.05)
    print("Diagnostic plot: fitted 0th order spec w/ cont pix")
    print("Saved as %s.png" % (figname))
    plt.savefig(figname)
    plt.close()


def diagnostics_contpix(self, data, nchunks=10, fig = "baseline_spec_with_cont_pix"):
    """ Call plot_contpix once for each nth of the spectrum """
    if data.contmask is None:
        print("No contmask set")
    else:
        coeffs_all = self.coeffs
        wl = data.wl
        baseline_spec = coeffs_all[: ,0]
        contmask = data.contmask
        contpix_x = wl[contmask]
        contpix_y = baseline_spec[contmask]
        rem = len(wl ) %nchunks
        wl_split = np.array(np.split(wl[0:len(wl ) -rem] ,nchunks))
        baseline_spec_split = np.array(
            np.split(baseline_spec[0:len(wl ) -rem] ,nchunks))
        nchunks = wl_split.shape[0]
        for i in range(nchunks):
            fig_chunk = fig + "_%s" %str(i)
            wl_chunk = wl_split[i, :]
            baseline_spec_chunk = baseline_spec_split[i, :]
            take = np.logical_and(
                contpix_x > wl_chunk[0], contpix_x < wl_chunk[-1])
            self.plot_contpix(
                wl_chunk, baseline_spec_chunk,
                contpix_x[take], contpix_y[take], fig_chunk)