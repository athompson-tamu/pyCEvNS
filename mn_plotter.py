import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.plot import *

import sys
import json


def main(mn_dir, filename, savename, nbins):

  paramsfile = open(mn_dir + "params.json")
  params = json.load(paramsfile)
  paramsfile.close()

  ndim = len(params)
  null_hyp = np.zeros(ndim)

  cp = CrediblePlot(mn_dir + filename)

  print("plotting grid")
  fig, ax = cp.credible_grid(params, null_hyp, params, nbins=nbins)
  png_str = "plots/credible/png/" + savename + ".png"
  pdf_str = "plots/credible/pdf/" + savename + ".pdf"
  fig.savefig(png_str)
  fig.savefig(pdf_str)

  return


if __name__ == "__main__":
  main(mn_dir=str(sys.argv[1]), filename=str(sys.argv[2]), savename=str(sys.argv[3]), nbins=40)
