import numpy as np
from scipy import stats
from scipy.optimize import fsolve, least_squares
import pandas as pd
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pynverse as pynv

from copula import MNStats

import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector
from rpy2.robjects import r
base = importr('base')
utils = importr('utils')
rcopula = importr('copula')





# Simulate bivariate pairs empirically.
class EmpricalCopula:
  def __init__(self, datastr):
    self.sample = np.genfromtxt(datastr)
    self.dim = self.sample.shape[1] - 1  # assume we take post-equal-weights, last entry is likelihood

  def simulate(self):
    pass


class EmpricalCopula12:
  def __init__(self):
    # read in table
    robjects.r('data = read.table(file = "multinest_posteriors/all_nsi_vector_solar_equal_weights.txt", header=F)')
    robjects.r('z = pobs(as.matrix(cbind(data[,1],data[,2])))')
  def simulate(self, u, v):
    def ddv(v2):
      #print("v2 = ", v2)
      v2 = float(v2)
      robjects.r('u = matrix(c({0}, {1}), 1, 2)'.format(u, v2))
      return np.asarray(robjects.r('dCn(u, U = z, j.ind = 1)'))
    try:
      return float(pynv.inversefunc(ddv, y_values=v, domain=[0, 1], open_domain=[True, True]))
    except:
      print("passing...")
      pass


class EmpricalCopula13:
  def __init__(self):
    # read in table
    robjects.r('data = read.table(file = "multinest_posteriors/all_nsi_vector_solar_equal_weights.txt", header=F)')
    robjects.r('z = pobs(as.matrix(cbind(data[,1],data[,3])))')
  def simulate(self, u, v):
    def ddv(v2):
      #print("v2 = ", v2)
      v2 = float(v2)
      robjects.r('u = matrix(c({0}, {1}), 1, 2)'.format(u, v2))
      return np.asarray(robjects.r('dCn(u, U = z, j.ind = 1)'))
    try:
      return float(pynv.inversefunc(ddv, y_values=v, domain=[0, 1], open_domain=[True, True]))
    except:
      print("passing...")
      pass





emp12 = EmpricalCopula12()
emp13 = EmpricalCopula13()


xee_post = np.genfromtxt("multinest_posteriors/all_nsi_vector_solar.txt")
xee_stats = MNStats(xee_post)


# Test hand-made Frank copula generator.
n_samples = 15000
r1 = np.empty(n_samples)
r2 = np.empty(n_samples)
r3 = np.empty(n_samples)
x1, p1 = xee_stats.GetMarginal(0)
x2, p2 = xee_stats.GetMarginal(1)
x3, p3 = xee_stats.GetMarginal(2)
cdf1 = np.cumsum(p1)
cdf2 = np.cumsum(p2)
cdf3 = np.cumsum(p3)

print("simulating copula...")
for i in range(0,n_samples):
  z = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
  u1 = z[0]
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    u2 = emp12.simulate(z[0], z[1])
    u3 = emp13.simulate(z[0], z[2])

  print(i)
  r1[i] = np.interp(u1, cdf1, x1)
  r2[i] = np.interp(u2, cdf2, x2)
  r3[i] = np.interp(u3, cdf3, x3)

sim12 = np.empty((r1.shape[0], 2))
sim13 = np.empty((r1.shape[0], 2))
sim23 = np.empty((r1.shape[0], 2))
sim12[:, 0] = r1
sim12[:, 1] = r2
sim13[:, 0] = r1
sim13[:, 1] = r3
sim23[:, 0] = r2
sim23[:, 1] = r3

np.savetxt("emp_copula_12_02.txt", sim12)
np.savetxt("emp_copula_13_02.txt", sim13)
np.savetxt("emp_copula_23_02.txt", sim23)
"""
sim12_01 = np.genfromtxt("emp_copula_12_01.txt")
sim12_02 = np.genfromtxt("emp_copula_12_02.txt")
#sim12_03 = np.genfromtxt("emp_copula_12_03.txt")
#sim12_04 = np.genfromtxt("emp_copula_12_04.txt")
sim12 = np.append(sim12_01, sim12_02, axis=0)
#sim_12 = np.append(sim_12, sim12_03, axis=0)
#sim_12 = np.append(sim_12, sim12_04, axis=0)

sim13_01 = np.genfromtxt("emp_copula_13_01.txt")
sim13_02 = np.genfromtxt("emp_copula_13_02.txt")
#sim13_03 = np.genfromtxt("emp_copula_13_03.txt")
#sim13_04 = np.genfromtxt("emp_copula_13_04.txt")
sim13 = np.append(sim13_01, sim13_02, axis=0)
#sim_13 = np.append(sim_13, sim13_03, axis=0)
#sim_13 = np.append(sim_13, sim13_04, axis=0)

sim23_01 = np.genfromtxt("emp_copula_23_01.txt")
sim23_02 = np.genfromtxt("emp_copula_23_02.txt")
#sim23_03 = np.genfromtxt("emp_copula_23_03.txt")
#sim23_04 = np.genfromtxt("emp_copula_23_04.txt")
sim23 = np.append(sim23_01, sim23_02, axis=0)
#sim_23 = np.append(sim_23, sim23_03, axis=0)
#sim_23 = np.append(sim_23, sim23_04, axis=0)


"""
sns.set_context("paper", font_scale = 1.8)
df12 = pd.DataFrame(sim12, columns=[r"$\epsilon^{e,V}_{ee}$", r"$\epsilon^{e,V}_{\mu\mu}$"])
df13 = pd.DataFrame(sim13, columns=[r"$\epsilon^{e,V}_{ee}$", r"$\epsilon^{e,V}_{\tau\tau}$"])
df23 = pd.DataFrame(sim23, columns=[r"$\epsilon^{e,V}_{\mu\mu}$", r"$\epsilon^{e,V}_{\tau\tau}$"])
sns_plot = sns.jointplot(x=r"$\epsilon^{e,V}_{ee}$", y=r"$\epsilon^{e,V}_{\mu\mu}$", data=df12,
                         kind="kde")
sns_plot.savefig("png/copula/xee/empirical_sim_ee_mm_xee.png")
sns_plot.savefig("pdf/copula/xee/empirical_sim_ee_mm_xee.pdf")

sns_plot = sns.jointplot(x=r"$\epsilon^{e,V}_{ee}$", y=r"$\epsilon^{e,V}_{\tau\tau}$", data=df13,
                         kind="kde")
sns_plot.savefig("png/copula/xee/empirical_sim_ee_tt_xee.png")
sns_plot.savefig("pdf/copula/xee/empirical_sim_ee_tt_xee.pdf")

sns_plot = sns.jointplot(x=r"$\epsilon^{e,V}_{\mu\mu}$", y=r"$\epsilon^{e,V}_{\tau\tau}$", data=df23,
                         kind="kde")
sns_plot.savefig("png/copula/xee/empirical_sim_mm_tt_xee.png")
sns_plot.savefig("pdf/copula/xee/empirical_sim_mm_tt_xee.pdf")


