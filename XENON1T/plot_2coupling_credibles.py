import numpy as np
from numpy import pi, log, log10, sqrt, exp
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from pyCEvNS.plot import *

from matplotlib.pylab import rc
import matplotlib.ticker as tickr
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import json

alpha = 1/137
me = 0.511e-3
def KSVZ(gg, e_by_n):
      cag = e_by_n - 1.92
      return 3*(1/(2*pi*cag))*alpha*me*gg*((e_by_n)*np.log(alpha*cag/2/pi/me/gg) - 1.92*log(1/me))

def DFSZI(ge):
      return ge/0.1944

def DFSZII(ge):
      return ge/0.108481


def CAST_07eV(ge):
      return 5e-20 / (2.5e-10+ge)

def CAST_10meV(ge):
      return 3e-22 / (ge+1.15e-12)

def main():

  nbins=40
  cl=(0.69,0.90)
  idx = (1, 0)


  cp = CrediblePlot("multinest/abc_loglog/abc_loglog.txt")
  cp_no_prim = CrediblePlot("multinest/abc_no_primakoff_loglog/abc_no_primakoff_loglog.txt")


  # Plot credible contours
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  cp.credible_2d(idx, credible_level=(0.70,0.95), nbins=nbins, ax=ax,
                            color='blue', alpha_range=(0.4,0.5))
  cp_no_prim.credible_2d(idx,credible_level=cl, nbins=nbins, ax=ax,
                            color='red', alpha_range=(0.4,0.5))
  # Plot QCD axion lines
  ge_list = np.logspace(-15,-11,1000)
  gg_list = np.logspace(-12,-7, 1000)
  support = np.ones(1000)
  plt.plot(log10(KSVZ(gg_list, 7/3)), log10(gg_list), color='k', ls='dashed')
  plt.plot(log10(KSVZ(gg_list, 44/3)), log10(gg_list), color='k', ls='dashed')
  plt.plot(log10(ge_list), log10(DFSZI(ge_list)), color='k', ls='-.')
  plt.plot(log10(ge_list), log10(DFSZII(ge_list)), color='k', ls='-.')

  # Plot astro bounds
  plt.plot(log10(ge_list), log10(0.66e-10 * support), color='darkred')
  plt.plot(log10(2.8e-13 * support), log10(gg_list), color='green')

  # Plot CAST lines
  #plt.plot(log10(ge_list), log10(CAST_07eV(ge_list)), color='gray', ls='dotted')
  #plt.plot(log10(ge_list), log10(CAST_10meV(ge_list)), color='gray', ls='solid')

  # Plot text
  text_fs = 12
  plt.text(log10(4e-13),-11.9,'DFSZ I', rotation=37, fontsize=text_fs, color="k", weight="bold")
  plt.text(-12.2,-11.13,'DFSZ II', rotation=37, fontsize=text_fs, color="k", weight="bold")
  plt.text(log10(4e-13),-8.3,r'KSVZ $E/N = 7/3$', rotation=38, fontsize=text_fs, color="k", weight="bold")
  plt.text(-14.0,-9.5,r'KSVZ $E/N = 44/3$', rotation=40, fontsize=text_fs, color="k", weight="bold")
  plt.text(-13.6,-10.4,"HB Stars", rotation=0, fontsize=text_fs, color="darkred", weight="bold")
  plt.text(-12.5,-10.7,'WDLF', rotation=0, fontsize=text_fs, color="green", weight="bold")

  # Plot arrows
  plt.arrow(log10(2.8e-13), -11, 0.08, 0, width=0.025, color='green')
  plt.arrow(-13.33, log10(0.66e-10), 0, 0.1, width=0.025, color='darkred')




  ax.set_ylabel(r"$\log_{10} (g_{a\gamma}$/GeV$^{-1}$)", fontsize=16)
  ax.set_xlabel(r"$\log_{10} g_{ae}$", fontsize=16)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)

  #plt.ylim((-12,-7.5))
  plt.xlim((-15,-11))
  #plt.xscale('log')

  import matplotlib.patches as patches
  rect1 = patches.Rectangle((0,0),1,1,facecolor='blue', alpha=0.5)
  rect2 = patches.Rectangle((0,0),1,1,facecolor='red', alpha=0.5)
  plt.legend((rect1, rect2), ('With Inverse Primakoff', 'Without Inverse Primakoff'),
             fontsize=10, framealpha=1.0)


  plt.tight_layout()
  plt.show()

  return


if __name__ == "__main__":
  main()
