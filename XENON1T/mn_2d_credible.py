import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from pyCEvNS.plot import *

from matplotlib.pylab import rc
import matplotlib.ticker as tickr
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import json

def DFSZ(ma):
        return (0.203*8/3 - 0.39)*ma*1e-9

def DFSZII(ma):
        return (0.203*2/3 - 0.39)*ma*1e-9

def KSVZ(ma, eByN):
        # ma in eV
        return (0.203*eByN - 0.39)*ma*1e-9



def main():

        nbins=100
        cl=(0.85,0.85)

        # Get tritium + Primakoff credibles
        cp = CrediblePlot("multinest/primakoff/primakoff.txt")
        cp_h3 = CrediblePlot("multinest/primakoff_h3/primakoff_h3.txt")
        cp_h3_prim = CrediblePlot("multinest/primakoff_h3_v2/primakoff_h3_v2.txt")

        # Get astro limits
        cast = np.genfromtxt("../data/existing_limits/cast.txt", delimiter=",")
        hbstars = np.genfromtxt("../data/existing_limits/hbstars.txt", delimiter=",")

        # Set colors
        from matplotlib.cm import get_cmap
        cmap = get_cmap('viridis')
        cmap_lines = get_cmap('inferno')
        color_primakoff= cmap(0.3)
        color_tritium = cmap(0.7)
        color_hbstars = cmap_lines(0.5)
        color_cast = cmap_lines(0.7)

        if False:
                # Plot tritium fit
                fig, ax = cp_h3_prim.credible_2d((1,0), credible_level=(0.68,0.95), nbins=nbins)
                plt.tight_layout()
                plt.show()
                plt.close()

                # Plot 1-D credibles
                fig, ax = cp.credible_1d(0, credible_level=(0.6827, 0.9545), nbins=80, ax=None,
                                give_max=False, label='', smooth=False, countour=True, give_edge=True,
                                color='b', ls='-', flip_axes=False, lwidth=2)
                cp_h3.credible_1d(0, credible_level=(0.6827, 0.9545), nbins=80, ax=ax,
                                give_max=False, label='', smooth=False, countour=True, give_edge=True,
                                color='r', ls='-', flip_axes=False, lwidth=2)
                plt.xlim((-11,-9.5))
                plt.tight_layout()
                plt.show()
                plt.close()

                


                # Plot credible contours
                fig, ax = cp_h3.credible_2d((1,0), credible_level=cl, nbins=nbins,
                                        color=color_tritium, alpha_range=(0.5,0.2))
                cp.credible_2d((1,0),credible_level=cl, nbins=nbins, color=color_primakoff,
                                ax=ax, alpha_range=(0.7,0.2))
                #ax.set_aspect('equal', adjustable='box')

                ax.plot(np.log10(hbstars[:,0]*1e6), np.log10(hbstars[:,1]*0.367e-3), label="HB Stars",
                        color=color_hbstars, ls='dashed', lw=2)
                ax.plot(np.log10(cast[:,0]*1e6), np.log10(cast[:,1]), label="CAST",
                        color=color_cast, ls='dashed', lw=2)
                ax.set_ylabel(r"$\log_{10} (g_{a\gamma\gamma}$ / GeV $^{-1}$)", fontsize=16)
                ax.set_xlabel(r"$\log_{10} (m_a$/keV)", fontsize=16)

                # Legend
                import matplotlib.patches as patches
                from matplotlib.lines import Line2D
                rect1 = patches.Rectangle((0,0),1,1,facecolor=color_primakoff)
                rect2 = patches.Rectangle((0,0),1,1,facecolor=color_tritium)
                hb_stars_line = Line2D([0], [0], color=color_hbstars, ls='dashed', lw=2)
                cast_line = Line2D([0], [0], color=color_cast, ls='dashed', lw=2)
                plt.legend((rect1,rect2,hb_stars_line,cast_line),
                        ('Primakoff', r'Primakoff + $^3$H',"HB Stars", "CAST"), loc="upper right",
                        fontsize=13, framealpha=1.0)
                plt.ylim((-11.5,-8.5))
                plt.xlim((-6,1))
                plt.tight_layout()
                plt.show()
                plt.close()


        # Plot smooth contours
        masses = np.logspace(-12, 2, 1000)
        support = np.ones_like(masses)

        #plt.plot(masses, np.power(10.0, -9.75)*support, color='crimson', label=r"Primakoff + $^3$H (2$\sigma$ Upper Limit)")
        
        plt.fill_between(masses, y1=np.power(10.0,-9.99) * support, y2=np.power(10.0,-9.49) * support,
                        facecolor='none', hatch='///', edgecolor='cadetblue', alpha=0.8, label=r"Primakoff + $^3$H (2$\sigma$ SR1 fit)")
        plt.fill_between(masses, y1=np.power(10.0,-9.87) * support, y2=np.power(10.0,-9.55) * support,
                         color='blue', alpha=0.8, label=r"Primakoff ($2\sigma$ SR1 fit)")
        
        # Plot 1ton exclusion
        exclusion_1ton = np.genfromtxt("data/XENON1T_exclusion.txt", delimiter=",")
        plt.plot(masses, 1.64e-10 * support,color='turquoise', lw=3, label=r"Primakoff ($2\sigma$ $B_0$ exclusion)")
        plt.arrow(4.35, 1.64e-10, 0, 5e-11, width=0.25, head_length=6e-11, head_width=0.6, color="turquoise")
        


        # External Limits
        plt.plot(hbstars[:,0]*1e9, 0.66*hbstars[:,1]*0.367e-3, color='darkred', ls='dotted', lw=2)
        plt.plot(cast[:,0]*1e9, cast[:,1], color='deeppink', ls='dashdot')
        plt.fill_between(masses, y1=KSVZ(masses, 44/3), y2=KSVZ(masses, 2), color='yellow', alpha=0.2)
        plt.plot(masses, DFSZ(masses), color='k', ls='dashed')

        # Text
        text_fs = 12
        plt.text(3e-2,1e-11,'DFSZ I', rotation=47, fontsize=text_fs, color="k", weight="bold")
        plt.text(1e-3,3e-12,r'KSVZ $E/N = 44/3$', rotation=47, fontsize=text_fs, color="k", weight="bold")
        plt.text(2e-1,3e-12,r'KSVZ $E/N = 2$', rotation=47, fontsize=text_fs, color="k", weight="bold")
        plt.text(1e1,3.5e-11,'HB Stars', rotation=0, fontsize=text_fs, color="darkred", weight="bold")
        plt.text(3e-4,3.5e-11,'CAST', rotation=0, fontsize=text_fs, color="deeppink", weight="bold")

        plt.ylabel(r"$g_{a\gamma}$ (GeV$^{-1}$)", fontsize=16)
        plt.xlabel(r"$m_a$ (eV)", fontsize=16)
        plt.xlim((1e-4, 1e2))
        plt.ylim((1e-12,1e-8))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(framealpha=1.0, loc="upper left", fontsize=12)
        plt.show()
        plt.close()





if __name__ == "__main__":
  main()
