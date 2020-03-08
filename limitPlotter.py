import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

import sys


def main(experiment, savename, labelname):
        beam = np.genfromtxt('data/beam.txt')
        eeinva = np.genfromtxt('data/eeinva.txt')
        lep = np.genfromtxt('data/lep.txt')
        nomad = np.genfromtxt('data/nomad.txt')

        dir = "limits/" + experiment + "/"


        #lower = np.genfromtxt("lower_limit_ar_coherent.txt")
        upper = np.genfromtxt(dir + "upper_limit.txt")
        removed = np.genfromtxt(dir + "removed_limit.txt")
        scatter = np.genfromtxt(dir + "scatter_limit.txt")
        scatter_trimmed = np.genfromtxt(dir + "scatter_limit_trimmed.txt")
        upper = np.flip(np.flip(upper[4:,:],axis=1))
        upper = upper[(upper[:,1] > 0)]
        fit = np.poly1d(np.polyfit(np.log10(upper[:,0]), np.log10(upper[:,1]), 1))


        total_limit = np.vstack((scatter[:-2,:], removed, upper))
        total_limit = total_limit[(total_limit[:,1] > 0)]
        total_limit = np.vstack((total_limit, [0.1, 10**fit(-1)]))


        fig, ax = plt.subplots()

        ax.plot(total_limit[:,0]*1e6, total_limit[:,1]*1e3,label=labelname)
        #plt.plot(upper[:,0]*1e6, 10**(fit(np.log10(upper[:,0])))*1e3, marker=".")

        #ax.plot(lower[:,0]*1e6, lower[:,1]*1e3, label="ext")
        #ax.plot(upper[:,0]*1e6, upper[:,1]*1e3, marker="o", label="hi")
        #ax.plot(removed[:,0]*1e6, removed[:,1]*1e3, marker="o", label="lo (r)", ls="--")
        #ax.plot(scatter[:-2,0]*1e6, scatter[:-2,1]*1e3, marker="o", label="scatter")
        #ax.plot(scatter_trimmed[:,0]*1e6, scatter_trimmed[:,1]*1e3, marker="o", label="scatter_trimmed", ls="--")

        ax.fill(beam[:,0], beam[:,1], label='Beam Dump', alpha=0.5)
        ax.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
                label=r'$e^+e^-\rightarrow inv.+\gamma$', alpha=0.5)
        ax.fill(lep[:,0], lep[:,1], label='LEP', alpha=0.5)
        ax.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
                label='NOMAD', alpha=0.5)
        ax.set_xlim(1,1e10)
        ax.set_ylim(10**(-8),1)
        ax.set_xlabel(r"$m_a$ [eV]")
        ax.set_ylabel('$g_{a\gamma\gamma}$ [GeV$^{-1}$]')
        plt.legend(loc="upper left", framealpha=1)
        plt.xscale('log')
        plt.yscale('log')

        png_str = "plots/ccm_limits/" + savename + ".png"
        pdf_str = "plots/ccm_limits/" + savename + ".pdf"

        plt.savefig(png_str)
        plt.savefig(pdf_str)


if __name__ == "__main__":
        main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
