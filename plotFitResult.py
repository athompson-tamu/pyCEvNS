import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.plot import *

import sys
import json

from matplotlib.pylab import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def main(mn_dir, filename, nbins=40, cl=(0.6827,0.6827)):

  paramsfile = open(mn_dir + "params.json")
  params = json.load(paramsfile)
  paramsfile.close()

  ndim = len(params)
  null_hyp = np.zeros(ndim)

  cp = CrediblePlot(mn_dir + filename)

  print("plotting grid")
  print(filename, mn_dir)
  fig, ax = cp.credible_2d((0,1), credible_level=cl, nbins=nbins)
  plt.xlabel(r"$\log_{10} m_V$", fontsize=20)
  plt.ylabel(r"$\log_{10} Y$", fontsize=20)
  plt.title(r"COHERENT CsI, $m_X = 75$ MeV, $m_\chi = 25$ MeV", loc="right")
  #plt.xscale('log')
  #plt.yscale('log')
  plt.show()

  return


if __name__ == "__main__":
  main(mn_dir=str(sys.argv[1]), filename=str(sys.argv[2]), nbins=50)
