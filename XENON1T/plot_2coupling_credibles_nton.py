import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, log, log10, sqrt, exp

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

def ExclusionCurve(log_ge):
      return -11/log_ge

def IAXO_03(ge):
      return 1e-22 / (ge)

def IAXO_01(ge):
      return 1e-24 / (ge)

iaxo_limit = np.genfromtxt("data/IAXO.txt", delimiter=",")


def main():

  if False:
      # Plot credible contours
      nbins=80
      cl=(0.95,0.95)
      idx = (1, 0)
      cp1 = CrediblePlot("multinest/abc_no_primakoff_exclusions/abc_no_primakoff_exclusions.txt")
      cp2 = CrediblePlot("multinest/abc_exclusions_1ton/abc_exclusions_1ton.txt")
      cp3 = CrediblePlot("multinest/abc_exclusions/abc_exclusions.txt")
      fig = plt.figure()
      ax = fig.add_subplot(1,1,1)
      
      cp1.credible_2d(idx, credible_level=(0.95,0.95), nbins=nbins, ax=ax,
                              color='blue', alpha_range=(0.4,0.5))
      cp2.credible_2d(idx, credible_level=(0.95,0.95), nbins=nbins, ax=ax,
                              color='green', alpha_range=(0.4,0.5))
      cp3.credible_2d(idx, credible_level=(0.95,0.95), nbins=nbins, ax=ax,
                              color='red', alpha_range=(0.4,0.5))
      plt.show()
      plt.close()

  exclusion = np.genfromtxt("data/XENONnT_exclusion.txt", delimiter=",")
  support = np.ones_like(exclusion[:,0])
  exclusion_no_prim = np.genfromtxt("data/XENONnT_exclusion_no_prim.txt", delimiter=",")
  exclusion_1ton = np.genfromtxt("data/XENON1T_exclusion.txt", delimiter=",")
  plt.fill_between(exclusion[:,0], exclusion[:,1], y2=np.ones_like(exclusion[:,0]), color='gray', edgecolor=(1,0,0,1), alpha=0.45)
  plt.plot(exclusion[:,0], exclusion[:,1], color='k', ls=":")
  plt.plot(exclusion_1ton[:,0], exclusion_1ton[:,1], color='darkorange', ls=':')
  plt.plot(exclusion_no_prim[:,0], exclusion_no_prim[:,1], color='crimson', ls=':')
  
  # Plot QCD axion lines
  ge_list = np.logspace(-15,-11,1000)
  gg_list = np.logspace(-15,-8, 1000)
  support = np.ones(1000)
  plt.plot(np.log10(KSVZ(gg_list, 7/3)), np.log10(gg_list), color='k', ls='dashed')
  plt.plot(np.log10(KSVZ(gg_list, 44/3)), np.log10(gg_list), color='k', ls='dashed')
  plt.plot(np.log10(ge_list), np.log10(DFSZI(ge_list)), color='k', ls='-.')
  plt.plot(np.log10(ge_list), np.log10(DFSZII(ge_list)), color='k', ls='-.')


  # Plot astro bounds
  plt.plot(np.log10(ge_list), np.log10(0.66e-10 * support), color='darkred')
  plt.plot(np.log10(2.8e-13 * support), np.log10(gg_list), color='green')

  plt.plot(np.log10(iaxo_limit[:,0]), np.log10(12.5*iaxo_limit[:,1]), color='royalblue', ls='solid',alpha=0.5)

  # Plot text
  text_fs = 12
  plt.text(-14.05,-10.5,r"IAXO+, $m_a = 0.2$eV", rotation=0, fontsize=10, color="royalblue", weight="bold")
  plt.text(-13.26,-12.1,'DFSZ II', rotation=23, fontsize=text_fs, color="k", weight="bold")
  plt.text(-13.26,-12.77,'DFSZ I', rotation=23, fontsize=text_fs, color="k", weight="bold")
  plt.text(-13.28,-9.3,r'KSVZ $E/N = 7/3$', rotation=26, fontsize=text_fs, color="k", weight="bold")
  plt.text(-13.4,-8.95,r'KSVZ $E/N = 44/3$', rotation=26, fontsize=text_fs, color="k", weight="bold")
  plt.text(-13.2,-10.4,"HB Stars", rotation=0, fontsize=text_fs, color="darkred", weight="bold")
  plt.text(-12.9,-11.07,'WDLF', rotation=0, fontsize=text_fs, color="green", weight="bold")
  plt.text(-14.95,-9.75,r"1 ton$\cdot$year" + "\n" + "(XENON1T)", rotation=0, fontsize=text_fs, color="darkorange", weight="bold")
  plt.text(-14.3,-10.8,r"1 kton$\cdot$year (G3 Xe)",
            rotation=0, fontsize=text_fs, color="k", weight="bold")
  plt.text(-14.7,-9.5,r"1 kton$\cdot$year (no I.P.)" + "\n" + "(G3 Xe)", rotation=-25, fontsize=text_fs, color="crimson", weight="bold")

  # Plot arrows
  plt.arrow(np.log10(2.8e-13), -11, 0.06, 0, width=0.05, head_width=0.15, head_length=0.03, color='green')
  #plt.arrow(-14.4, np.log10(0.66e-10), 0, 0.1, width=0.025, color='darkred')
  plt.arrow(-13, np.log10(0.66e-10), 0, 0.1, width=0.025, color='darkred')




  plt.ylabel(r"$\log_{10} (g_{a\gamma}$/GeV$^{-1}$)", fontsize=16)
  plt.xlabel(r"$\log_{10} g_{ae}$", fontsize=16)
  plt.xticks(fontsize=13)
  plt.yticks(fontsize=13)

  plt.ylim((-13,-8))
  plt.xlim((-15,-12))
  #plt.xscale('log')

  #plt.axes().set_aspect('equal')
  plt.tight_layout()
  plt.show()

  return


if __name__ == "__main__":
  main()
