import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.plot import *

import sys
import json


def main(mn_dir, filename, nbins=40, cl=(0.6827,0.95), idx1=0, idx2=1):

  paramsfile = open(mn_dir + "params.json")
  params = json.load(paramsfile)
  paramsfile.close()

  ndim = len(params)
  null_hyp = np.zeros(ndim)

  cp = CrediblePlot(mn_dir + filename)
  idx = [idx1, idx2]

  print("plotting grid")
  print(filename, mn_dir)
  fig, ax = cp.credible_2d((1,0), credible_level=cl, nbins=nbins)
  ax.set_ylabel(r"$\log\epsilon^2$", fontsize=15)
  ax.set_xlabel(r"$\log \mathcal{N}$", fontsize=15)
  ax.set_title(r"$\mathcal{E} = \mathcal{N} \times \mathcal{E}_{CCM}$", loc="right")
  plt.show()

  return


if __name__ == "__main__":
  main(mn_dir=str(sys.argv[1]), filename=str(sys.argv[2]), nbins=40, idx1=sys.argv[3], idx2=sys.argv[4])
