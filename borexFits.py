import numpy as np
from scipy.integrate import quad


# Embed the Borexino background fits
class BorexFit:
  def __init__(self, bkg):
    if bkg == "po":
      self.params = np.array([6.76,439.6,40.03])
    if bkg == "bi":
      self.params = np.array([0.0627,101.8,401.2])
    if bkg == "kr":
      self.params = np.array([0.051,200.36,205])
    if bkg == "c11":
      self.params = np.array([0.01056,1284.67,203.5])
    if bkg == "c14":
      self.params = np.array([16995.7,72.19,30.73])

  def gaus(self, x):
    a = self.params[0]
    x0 = self.params[1]
    sigma = self.params[2]
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

  # Return the number of events per 100ton - day
  def events(self, low, high):
    return quad(self.gaus, low, high)[0]