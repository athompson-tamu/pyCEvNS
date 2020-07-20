import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.plot import *

import sys
import json


def main(mn_dir, filename, nbins):

  paramsfile = open(mn_dir + "params.json")
  params = json.load(paramsfile)
  paramsfile.close()

  ndim = len(params)
  #null_hyp = np.zeros(ndim)
  null_hyp = [0.5, 0.0, 0.0, 0.5, 0.5, 0.5]
  null_hyp = [1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]
  cp = CrediblePlot(mn_dir + filename)

  print("plotting grid")
  fig, ax = cp.credible_grid(params, null_hyp, params, nbins=nbins)
  #png_str = "plots/credible/png/" + savename + ".png"
  #pdf_str = "plots/credible/pdf/" + savename + ".pdf"
  #fig.savefig(png_str)
  #fig.savefig(pdf_str)
  plt.show()

  return


if __name__ == "__main__":
  main(mn_dir=str(sys.argv[1]), filename=str(sys.argv[2]), nbins=40)
