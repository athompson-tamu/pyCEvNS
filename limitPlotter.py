import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
from scipy.ndimage import median_filter
from scipy.ndimage.filters import gaussian_filter1d, convolve
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, pi
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# get existing limits
relic = np.genfromtxt('pyCEvNS/data/dark_photon_limits/relic.txt', delimiter=",")
lsnd = np.genfromtxt('pyCEvNS/data/dark_photon_limits/lsnd.txt', delimiter=",")
babar = np.genfromtxt('pyCEvNS/data/dark_photon_limits/babar.txt', delimiter=",")
miniboone_e = np.genfromtxt('pyCEvNS/data/dark_photon_limits/miniboone-e.txt', delimiter=",")
miniboone_n = np.genfromtxt('pyCEvNS/data/dark_photon_limits/miniboone-n.txt', delimiter=",")
na64 = np.genfromtxt('pyCEvNS/data/dark_photon_limits/na64.txt', delimiter=",")

# COHERENT 610 projection
lar610 = np.genfromtxt('limits/coherent/projection_lar_610_1912-06422.txt', delimiter=",")


# Convert from GeV mass to MeV
lsnd[:,0] *= 1000
miniboone_e[:,0] *= 1000
miniboone_n[:,0] *= 1000
na64[:,0] *= 1000
babar[:,0] *= 1000
lar610[:,0] *= 3


# Rescale the limits
eps_rescale = (4*pi/137) / ((3**4) * 4*pi*np.sqrt(10/137))
y_rescale = 1 #(3**4) * 4*pi*np.sqrt(10/137)

#ldmx[:,1] *= 2 * (3**4)
#lar610[:,1] *= (3**4) * 4*pi*np.sqrt(2/137)
relic[:,1] *= eps_rescale
na64[:,1] *= y_rescale
lsnd[:,1] *= y_rescale
miniboone_e[:,1] *= y_rescale
miniboone_n[:,1] *= y_rescale
babar[:,1] *= y_rescale

# Work out text positions.
def get_text_pos(array, mass):
        return np.interp(10, array[:,0], array[:,1])


def smoother(arr):
    grad = convolve(arr, [1,-1,0], mode="nearest")[:-1]
    smooth_grad = smooth_grad = gaussian_filter1d(grad, 0.25)
    # Integrate
    smoothed = [arr[0] + sum(smooth_grad[:x]) for x in range(len(arr))]
    return smoothed



def plot_singlemed():
        # Plot single-mediator limits
        
        # Load derived limits.
        ccm_loose = np.genfromtxt("limits/ccm/dark_photon_limits_singlemed_ccm_loose_100pts.txt", delimiter=",")
        ccm_tight = np.genfromtxt("limits/ccm/dark_photon_limits_singlemed_ccm_tight.txt", delimiter=",")
        coherent = np.genfromtxt("limits/coherent/dark_photon_limits_coh_singlemed_csi-lar.txt", delimiter=",")
        coherent_futureLAr = np.genfromtxt("limits/coherent/dark_photon_limits_coh_singlemed_futureLAr.txt", delimiter=",")
        jsns2 = np.genfromtxt("limits/jsns2/dark_photon_limits_jsns_singlemed_withEta.txt", delimiter=",")

        # Plot the existing limits.
        plt.fill_between(babar[:,0], babar[:,1], y2=1, color="teal", alpha=0.15)
        plt.fill_between(miniboone_n[:,0], miniboone_n[:,1], y2=1, color="mediumpurple", alpha=0.15)
        plt.fill_between(na64[:,0], na64[:,1], y2=1, color="tan", alpha=0.15)
        plt.fill_between(miniboone_e[:,0], miniboone_e[:,1], y2=1, color="orchid", alpha=0.15)
        plt.fill_between(lsnd[:,0], lsnd[:,1], y2=1, color="chocolate", alpha=0.15)

        plt.plot(babar[:,0], babar[:,1], color="teal", ls="dashed")
        plt.plot(miniboone_n[:,0], miniboone_n[:,1], color="mediumpurple", ls="dashed")
        plt.plot(na64[:,0], na64[:,1], color="tan", ls="dashed")
        plt.plot(miniboone_e[:,0], miniboone_e[:,1], color="orchid", ls="dashed")
        plt.plot(lsnd[:,0], lsnd[:,1], color="chocolate", ls="dashed")

        # Plot relic density limit
        plt.plot(relic[:,0], relic[:,1], color="k", linewidth=2)
        
        # Smooth out the limits
        jsns2[:,1] = signal.savgol_filter(jsns2[:,1], 13, 2)
        
        masses_fine = np.logspace(np.log10(ccm_tight[0,0]), np.log10(ccm_tight[-1,0]), 1000)

        # Plot the derived limits
        plt.plot(ccm_tight[:,0], eps_rescale*ccm_tight[:,1], label="CCM LAr (Tight WP)", linewidth=2, color="dodgerblue")
        plt.plot(ccm_loose[:,0], eps_rescale*ccm_loose[:,1], label="CCM LAr (Loose WP)", linewidth=2, ls='dashed', color="dodgerblue")
        plt.plot(coherent[:,0], eps_rescale*coherent[:,1], label="COHERENT CsI + LAr", linewidth=2, color="crimson")
        plt.plot(coherent_futureLAr[:,0], eps_rescale*coherent_futureLAr[:,1], label="COHERENT Future-LAr (this work)", linewidth=2, ls='dashed', color="crimson")
        # Plot the lar610 projection
        plt.plot(lar610[:,0], lar610[:,1], color="darkred", ls="dashdot", linewidth=2, label="COHERENT Future-LAr (1912.06422)")
        plt.plot(jsns2[:,0], eps_rescale*jsns2[:,1], label=r"JSNS$^2$", linewidth=2, color="orange")
        
        
        # Draw text for existing limits.
        text_fs = 13
        plt.text(6,4e-11,'MiniBooNE \n (Nucleus)', rotation=0, fontsize=text_fs, color="mediumpurple", weight="bold")
        plt.text(100,1.5e-7,'MiniBooNE \n (Electron)', rotation=0, fontsize=text_fs, color="orchid", weight="bold")
        plt.text(40,1e-10,'NA64', rotation=25, fontsize=text_fs, color="tan", weight="bold")
        plt.text(20,1.5e-10,'LSND', rotation=20, fontsize=text_fs, color="chocolate", weight="bold")
        plt.text(15,2e-9,'BaBar', rotation=0, fontsize=text_fs, color="teal", weight="bold")
        plt.text(100,2e-13,'Relic Density', rotation=20, fontsize=text_fs, color="k", weight="bold")
        
        plt.title(r"$m_V = m_X = 3m_\chi$", loc='right', fontsize=15)
        plt.legend(loc="upper left", fontsize=9, framealpha=1.0)

        plt.xscale("Log")
        plt.yscale("Log")
        plt.xlim((3, 455))
        plt.ylim((1e-15,2e-5))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylabel(r"$Y\equiv \epsilon^2 \alpha_D (\frac{m_\chi}{m_X})^4$", fontsize=15)
        plt.xlabel(r"$m_X$ [MeV]", fontsize=15)
        plt.tight_layout()
        plt.savefig("paper_plots/combined_limits_singlemed.png")
        plt.show()
        plt.clf()


def plot_doublemed():
        epsilon = 0.01**2
        
        # Double-mediator
        ccm_loose = np.genfromtxt("limits/ccm/dark_photon_limits_doublemed_ccm_loose.txt", delimiter=",")
        ccm_tight = np.genfromtxt("limits/ccm/dark_photon_limits_doublemed_ccm_tight.txt", delimiter=",")
        coherent = np.genfromtxt("limits/coherent/dark_photon_limits_coh_doublemed_csi-lar.txt", delimiter=",")
        coherent_futureLAr = np.genfromtxt("limits/coherent/dark_photon_limits_coh_doublemed_futureLAr.txt", delimiter=",")
        jsns2 = np.genfromtxt("limits/jsns2/dark_photon_limits_jsns_doublemed.txt", delimiter=",")
        
        coherent_futureLAr[:,1] *= (685/610)

        # Plot the existing limits.
        plt.fill_between(babar[:,0], epsilon*babar[:,1], y2=1, color="teal", alpha=0.15)
        plt.fill_between(miniboone_n[:,0], epsilon*miniboone_n[:,1], y2=1, color="mediumpurple", alpha=0.15)
        plt.fill_between(na64[:,0], epsilon*na64[:,1], y2=1, color="tan", alpha=0.15)
        plt.fill_between(miniboone_e[:,0], epsilon*miniboone_e[:,1], y2=1, color="orchid", alpha=0.15)
        plt.fill_between(lsnd[:,0], epsilon*lsnd[:,1], y2=1, color="chocolate", alpha=0.15)

        plt.plot(babar[:,0], epsilon*babar[:,1], color="teal", ls="dashed")
        plt.plot(miniboone_n[:,0], epsilon*miniboone_n[:,1], color="mediumpurple", ls="dashed")
        plt.plot(na64[:,0], epsilon*na64[:,1], color="tan", ls="dashed")
        plt.plot(miniboone_e[:,0], epsilon*miniboone_e[:,1], color="orchid", ls="dashed")
        plt.plot(lsnd[:,0], epsilon*lsnd[:,1], color="chocolate", ls="dashed")

        # Plot relic density limit
        #plt.plot(relic[:,0], relic[:,1], color="k", linewidth=2)

        # Plot the derived limits
        plt.plot(ccm_tight[:,0], epsilon*eps_rescale*ccm_tight[:,1], label="CCM LAr (Tight WP)", linewidth=2, color="dodgerblue")
        plt.plot(ccm_loose[:,0], epsilon*eps_rescale*ccm_loose[:,1], label="CCM LAr (Loose WP)", linewidth=2, ls='dashed', color="dodgerblue")
        plt.plot(coherent[:,0], epsilon*eps_rescale*coherent[:,1], label="COHERENT CsI + LAr", linewidth=2, color="crimson")
        plt.plot(coherent_futureLAr[:,0], epsilon*eps_rescale*coherent_futureLAr[:,1], label="COHERENT Future-LAr", linewidth=2, ls='dashed', color="crimson")
        plt.plot(jsns2[:,0], epsilon*eps_rescale*jsns2[:,1], label=r"JSNS$^2$", linewidth=2, color="orange")
        
        # Draw text for existing limits.
        text_fs = 13
        plt.text(6,epsilon*4e-11,'MiniBooNE \n (Nucleus)', rotation=0, fontsize=text_fs, color="mediumpurple", weight="bold")
        plt.text(50,epsilon*2e-8,'MiniBooNE \n (Electron)', rotation=0, fontsize=text_fs, color="orchid", weight="bold")
        plt.text(40,epsilon*1e-10,'NA64', rotation=25, fontsize=text_fs, color="tan", weight="bold")
        plt.text(20,epsilon*1.5e-10,'LSND', rotation=20, fontsize=text_fs, color="chocolate", weight="bold")
        plt.text(138,epsilon*2e-9,'BaBar', rotation=0, fontsize=text_fs, color="teal", weight="bold")
        #plt.text(100,2e-13,'Relic Density', rotation=20, fontsize=text_fs, color="k", weight="bold")
        
        plt.title(r"$\epsilon = 0.01$, $m_X=75$ MeV, $m_\chi =25$ MeV", loc='right', fontsize=15)
        plt.legend(loc="upper left", fontsize=9, framealpha=1.0)

        plt.xscale("Log")
        plt.yscale("Log")
        plt.xlim((3, 450))
        plt.ylim((epsilon*1e-15,epsilon*2e-7))
        plt.ylabel(r"$Y\equiv \epsilon^2 \alpha_D (\frac{m_\chi}{m_X})^4$", fontsize=15)
        plt.xlabel(r"$m_V$ [MeV]", fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.tight_layout()
        plt.savefig("paper_plots/combined_limits_doublemed.png")
        plt.show()
        plt.clf()


if __name__ == "__main__":
        plot_singlemed()
        plot_doublemed()
