import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.plot import *

import sys
import json


def main(mn_dir, filename, nbins=40, cl=(0.6827,)):

  paramsfile = open(mn_dir + "params.json")
  params = json.load(paramsfile)
  paramsfile.close()

  ndim = len(params)
  null_hyp = np.zeros(ndim)

  cp = CrediblePlot(mn_dir + filename)

  print("plotting grid")
  print(filename, mn_dir)
  fig, ax = cp.credible_grid(params, null_hyp, params, credible_level=cl, nbins=nbins)
  plt.show()

  return


if __name__ == "__main__":
  main(mn_dir=str(sys.argv[1]), filename=str(sys.argv[2]), nbins=40)
