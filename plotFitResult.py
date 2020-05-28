import numpy as np
from numpy import sqrt, pi
import matplotlib.pyplot as plt

from pyCEvNS.plot import *

import sys
import json

from matplotlib.pylab import rc

import matplotlib.patches as patches





rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

echarge = np.sqrt(4*np.pi/137)
def ConvertCoupling(file):
  data = np.genfromtxt(file)
  eps2Y = sqrt(4*pi*0.5) / (3**4) / 4 / np.pi / echarge
  data[:,3] = np.log10(np.power(10,data[:,3]) * eps2Y)
  return data
  



def main(mn_dir, filename, filename2, cl=(0.6875,0.6875)):
  nbins = 100

  paramsfile = open(mn_dir + "params.json")
  params = json.load(paramsfile)
  paramsfile.close()

  ndim = len(params)
  null_hyp = np.zeros(ndim)

  y_scaled_data = ConvertCoupling(mn_dir + filename)
  cp = CrediblePlot(y_scaled_data)
  
  # second fit
  cp2 = CrediblePlot(ConvertCoupling(mn_dir + filename2))

  print("plotting grid")
  print(filename, mn_dir)
  fig, ax = cp.credible_2d((0,1), credible_level=cl, nbins=nbins, color='blue')
  cp2.credible_2d((0,1), credible_level=cl, nbins=nbins, ax=ax, color='red')
  plt.xlabel(r"$\log_{10} m_V$", fontsize=20)
  plt.ylabel(r"$\log_{10} Y$", fontsize=20)
  plt.title(r"COHERENT CsI, $m_V = m_X = 3 m_\chi$", loc="right")
  
  rect1 = patches.Rectangle((0,0),1,1,facecolor='royalblue')
  rect2 = patches.Rectangle((0,0),1,1,facecolor='lightcoral')
  plt.legend((rect1, rect2), ('With Cuts', 'Without Cuts'), loc="upper left",
             fontsize=15, framealpha=1.0)
  #plt.xscale('log')
  #plt.yscale('log')
  plt.ylim((-11, -8))
  plt.xlim((np.log10(3), np.log10(300)))
  plt.tight_layout()
  plt.show()

  return


if __name__ == "__main__":
  main(mn_dir=str(sys.argv[1]), filename=str(sys.argv[2]),filename2=str(sys.argv[3]))
