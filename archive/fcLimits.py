"""
Analytical solution for Poisson process with background.

Produces Fig. 7 from the Feldman Cousins paper.
"""
from functools import partial
import numpy as np
from astropy.utils.console import ProgressBar
import matplotlib.pyplot as plt
from gammapy.stats import fc_find_acceptance_interval_poisson, fc_fix_limits

if __name__ == "__main__":


    dru_bkg = 100
    mass = 4
    days = 3 * 365
    background = dru_bkg * mass * days

    n_bins_x = 2 * background
    step_width_mu = 0.005 * background
    mu_min = 0.05 * background
    mu_max = 1.5*background
    cl = 0.90

    x_bins = np.arange(int(0.01 * background), n_bins_x, 10)
    mu_bins = np.linspace(
        mu_min, mu_max, int(mu_max / step_width_mu) + 1, endpoint=True
    )

    print("Generating FC confidence belt for %s values of mu." % len(mu_bins))

    partial_func = partial(
        fc_find_acceptance_interval_poisson,
        background=background,
        x_bins=x_bins,
        alpha=cl,
    )

    results = ProgressBar.map(partial_func, mu_bins, multiprocess=True)

    LowerLimitAna, UpperLimitAna = zip(*results)

    LowerLimitAna = np.asarray(LowerLimitAna)
    UpperLimitAna = np.asarray(UpperLimitAna)

    fc_fix_limits(LowerLimitAna, UpperLimitAna)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(LowerLimitAna, mu_bins, ls="-", color="red")
    plt.plot(UpperLimitAna, mu_bins, ls="-", color="red")

    plt.grid(True)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.xticks(range(15))
    plt.yticks(range(15))
    ax.set_xlabel(r"Measured n")
    ax.set_ylabel(r"Signal Mean $\mu$")
    plt.axis([0, 15, 0, 15])
    plt.savefig("feldman_cousins/example.png")
    #plt.show()
