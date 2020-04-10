import sys

from pyCEvNS.events import *
from pyCEvNS.flux import *

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, pi
from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_singlemed():
        # Plot single-mediator limits
        # get existing limits
        relic = np.genfromtxt('pyCEvNS/data/dark_photon_limits/relic.txt', delimiter=",")
        ldmx = np.genfromtxt('pyCEvNS/data/dark_photon_limits/ldmx.txt')
        lsnd = np.genfromtxt('pyCEvNS/data/dark_photon_limits/lsnd.csv', delimiter=",")
        miniboone = np.genfromtxt('pyCEvNS/data/dark_photon_limits/miniboone.csv', delimiter=",")
        na64 = np.genfromtxt('pyCEvNS/data/dark_photon_limits/na64.csv', delimiter=",")

        # Convert from GeV mass to MeV
        lsnd[:,0] *= 1000
        miniboone[:,0] *= 1000
        miniboone[:,2] *= 1000
        na64[:,0] *= 1000

        #ldmx[:,1] *= 2 * (3**4)
        na64[:,1] *= 4*pi * (3**4)
        lsnd[:,1] *= 4*pi * (3**4)  # TODO: check this
        miniboone[:,1] *= 4*pi * (3**4)
        miniboone[:,3] *= 4*pi * (3**4)
        
        # Load derived limits.
        ccm_loose = np.genfromtxt("limits/ccm/dark_photon_limits_ccm_loose.txt", delimiter=",")
        ccm_tight = np.genfromtxt("limits/ccm/dark_photon_limits_ccm_tight.txt", delimiter=",")
        coherent = np.genfromtxt("limits/coherent/dark_photon_limits_coh_singlemed_csi-lar.txt", delimiter=",")
        coherent_futureLAr = np.genfromtxt("limits/coherent/dark_photon_limits_coh_singlemed_futureLAr.txt", delimiter=",")
        jsns2 = np.genfromtxt("limits/jsns2/dark_photon_limits_jsns_singlemed.txt", delimiter=",")


        # Plot the existing limits.
        plt.fill_between(miniboone[:,0], miniboone[:,1], y2=1, color="royalblue", alpha=0.3, label='MiniBooNE \n (Nucleus)')
        plt.fill_between(na64[:,0], na64[:,1], y2=1, color="maroon", alpha=0.3, label='NA64')
        plt.fill_between(miniboone[:,2], miniboone[:,3], y2=1, color="orchid", alpha=0.3, label='MiniBooNE \n (Electron)')
        plt.fill_between(lsnd[:,0], lsnd[:,1], y2=1, color="crimson", alpha=0.3, label='LSND')

        plt.plot(miniboone[:,0], miniboone[:,1], color="royalblue", ls="dashed")
        plt.plot(na64[:,0], na64[:,1], color="maroon", ls="dashed")
        plt.plot(miniboone[:,2], miniboone[:,3], color="orchid", ls="dashed")
        plt.plot(lsnd[:,0], lsnd[:,1], color="crimson", ls="dashed")

        # Plot relic density limit
        plt.plot(relic[:,0], relic[:,1], color="k", linewidth=2, label="Relic Density")

        # Plot the derived limits
        plt.plot(ccm_tight[:,0], ccm_tight[:,1], label="CCM LAr (Tight WP)", linewidth=2, color="dodgerblue")
        plt.plot(ccm_loose[:,0], ccm_loose[:,1], label="CCM LAr (Loose WP)", linewidth=2, ls='dashed', color="dodgerblue")
        plt.plot(coherent[:,0], coherent[:,1], label="COHERENT CsI + LAr", linewidth=2, color="crimson")
        plt.plot(coherent_futureLAr[:,0], coherent_futureLAr[:,1], label="COHERENT Future-LAr", linewidth=2, ls='dashed', color="crimson")
        plt.plot(jsns2[:,0], jsns2[:,1], label=r"JSNS$^2$", linewidth=2, color="orange")
        
        plt.title(r"Single-mediator scenario; $m_V = m_X = 3m_\chi$ MeV", loc='right', fontsize=15)
        plt.legend(loc="upper left", ncol=2, fontsize=10, framealpha=1.0)

        plt.xscale("Log")
        plt.yscale("Log")
        plt.xlim((1, 5e2))
        plt.ylim((1e-11,2e-4))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylabel(r"$\epsilon \kappa^X_f \kappa^X_D$", fontsize=15)
        plt.xlabel(r"$m_X$ [MeV]", fontsize=15)
        plt.tight_layout()
        plt.savefig("paper_plots/combined_limits_singlemed.png")
        plt.show()
        plt.clf()


def plot_doublemed():
        # Plot single-mediator limits
        # get existing limits
        relic = np.genfromtxt('pyCEvNS/data/dark_photon_limits/relic.txt', delimiter=",")
        ldmx = np.genfromtxt('pyCEvNS/data/dark_photon_limits/ldmx.txt')
        lsnd = np.genfromtxt('pyCEvNS/data/dark_photon_limits/lsnd.csv', delimiter=",")
        miniboone = np.genfromtxt('pyCEvNS/data/dark_photon_limits/miniboone.csv', delimiter=",")
        na64 = np.genfromtxt('pyCEvNS/data/dark_photon_limits/na64.csv', delimiter=",")

        # Convert from GeV mass to MeV
        lsnd[:,0] *= 1000
        miniboone[:,0] *= 1000
        miniboone[:,2] *= 1000
        na64[:,0] *= 1000

        #ldmx[:,1] *= 2 * (3**4)
        na64[:,1] *= 4*pi * (3**4)
        lsnd[:,1] *= 4*pi * (3**4)  # TODO: check this
        miniboone[:,1] *= 4*pi * (3**4)
        miniboone[:,3] *= 4*pi * (3**4)
        
        # Double-mediator
        ccm_loose = np.genfromtxt("limits/ccm/dark_photon_limits_doublemed_ccm_loose.txt", delimiter=",")
        ccm_tight = np.genfromtxt("limits/ccm/dark_photon_limits_doublemed_ccm_tight.txt", delimiter=",")
        coherent = np.genfromtxt("limits/coherent/dark_photon_limits_coh_doublemed_csi-lar.txt", delimiter=",")
        coherent_futureLAr = np.genfromtxt("limits/coherent/dark_photon_limits_coh_doublemed_futureLAr.txt", delimiter=",")
        jsns2 = np.genfromtxt("limits/jsns2/dark_photon_limits_jsns_doublemed.txt", delimiter=",")


        # Plot the existing limits.
        plt.fill_between(miniboone[:,0], miniboone[:,1], y2=1, color="royalblue", alpha=0.3, label='MiniBooNE \n (Nucleus)')
        plt.fill_between(na64[:,0], na64[:,1], y2=1, color="maroon", alpha=0.3, label='NA64')
        plt.fill_between(miniboone[:,2], miniboone[:,3], y2=1, color="orchid", alpha=0.3, label='MiniBooNE \n (Electron)')
        plt.fill_between(lsnd[:,0], lsnd[:,1], y2=1, color="crimson", alpha=0.3, label='LSND')

        plt.plot(miniboone[:,0], miniboone[:,1], color="royalblue", ls="dashed")
        plt.plot(na64[:,0], na64[:,1], color="maroon", ls="dashed")
        plt.plot(miniboone[:,2], miniboone[:,3], color="orchid", ls="dashed")
        plt.plot(lsnd[:,0], lsnd[:,1], color="crimson", ls="dashed")

        # Plot relic density limit
        plt.plot(relic[:,0], relic[:,1], color="k", linewidth=2, label="Relic Density")

        # Plot the derived limits
        plt.plot(ccm_tight[:,0], ccm_tight[:,1], label="CCM LAr (Tight WP)", linewidth=2, color="dodgerblue")
        plt.plot(ccm_loose[:,0], ccm_loose[:,1], label="CCM LAr (Loose WP)", linewidth=2, ls='dashed', color="dodgerblue")
        plt.plot(coherent[:,0], coherent[:,1], label="COHERENT CsI + LAr", linewidth=2, color="crimson")
        plt.plot(coherent_futureLAr[:,0], coherent_futureLAr[:,1], label="COHERENT Future-LAr", linewidth=2, ls='dashed', color="crimson")
        plt.plot(jsns2[:,0], jsns2[:,1], label=r"JSNS$^2$", linewidth=2, color="orange")
        
        plt.title(r"Double-mediator scenario; $m_X=75$ MeV, $m_\chi =25$ MeV", loc='right', fontsize=15)
        plt.legend(loc="upper left", ncol=2, fontsize=10, framealpha=1.0)

        plt.xscale("Log")
        plt.yscale("Log")
        plt.xlim((1, 5e2))
        plt.ylim((1e-11,2e-4))
        plt.ylabel(r"$\epsilon \kappa^V_f \kappa^V_D$", fontsize=15)
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
