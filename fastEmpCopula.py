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

from matplotlib.pylab import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



class EmpricalCopula:
    def __init__(self, datastr, i, j):
        # read in table
        robjects.r('data = read.table(file = "{0}", header=F)'.format(datastr))
        robjects.r('z = pobs(as.matrix(cbind(data[,{0}],data[,{1}])))'.format(i, j))
    def simulate(self, u, v):
        def ddv(v2):
            v2 = float(v2)
            robjects.r('u = matrix(c({0}, {1}), 1, 2)'.format(u, v2))
            return np.asarray(robjects.r('dCn(u, U = z, j.ind = 1)'))
        try:
            return float(pynv.inversefunc(ddv, y_values=v, domain=[0, 1], open_domain=[True, True],
                                        image=[0,1], accuracy=1))
        except:
        #print("passing...")
            return v


class FastEmpricalCopula:
    def __init__(self, datastr, i, j):
        # read in table
        robjects.r('data = read.table(file = "{0}", header=F)'.format(datastr))
        robjects.r('z = pobs(as.matrix(cbind(data[,{0}],data[,{1}])))'.format(i, j))
    
    def ddv(self, u, v2):
            robjects.r('u = matrix(c({0}, {1}), 1, 2)'.format(u, v2))
            return np.asarray(robjects.r('dCn(u, U = z, j.ind = 1)'))[0]
    
    def simulate(self, u, v):
        v_list = np.linspace(0,1,25)
        ddv_list = [self.ddv(u, _v) for _v in v_list]
        return np.interp(v, ddv_list, v_list)
        

def FrankCopula(u1, v2, theta):
    u2 = - (1/theta) * np.log(1 + (v2 * (1 - np.exp(-theta))) / (v2 * (np.exp(-theta * u1) - 1) -
                                                                        np.exp(-theta * u1)))
    return u2



# Get indices
idx1 = 0 # 2
idx2 = 1 # 8

# File strings
data_str = "multinest_jhep/all_nsi_vector_solar.txt"
wgts_str = "multinest_jhep/all_nsi_vector_solar_equal_weights.txt"
#data_str = "multinest_jhep/all_nsi_xenon_atmos_coh_prior_jhep.txt"
#wgts_str = "multinest_jhep/all_nsi_xenon_atmos_coh_prior_jheppost_equal_weight.txt"

# Get empirical copula
emp_ee_mm = FastEmpricalCopula(wgts_str, idx1+1, idx2+1)


# Get observations
xee_post = np.genfromtxt(data_str)
xee_stats = MNStats(xee_post)

data = np.genfromtxt(wgts_str)
data_12 = np.empty((data.shape[0], 2))
data_12[:, 0] = data[:,idx1]
data_12[:, 1] = data[:,idx2]


# Test hand-made Frank copula generator.
n_samples = 100
r1 = np.empty(n_samples)
r2 = np.empty(n_samples)
frank1 = np.empty(n_samples)
frank2 = np.empty(n_samples)
x1, p1 = xee_stats.GetMarginal(idx1)
x2, p2 = xee_stats.GetMarginal(idx2)
cdf1 = np.cumsum(p1)
cdf2 = np.cumsum(p2)

print("simulating copula...")
for i in range(0,n_samples):
  z = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
  # Empirical
  u1 = z[0]
  u2 = emp_ee_mm.simulate(z[0], z[1])

  # Frank
  u1_fr = z[0]
  u2_fr = FrankCopula(u1_fr, z[1], theta=-14.0)

  print(i)
  r1[i] = np.interp(u1, cdf1, x1)
  r2[i] = np.interp(u2, cdf2, x2)
  frank1[i] = np.interp(u1_fr, cdf1, x1)
  frank2[i] = np.interp(u2_fr, cdf2, x2)

sim12 = np.empty((r1.shape[0], 2))
sim12[:, 0] = r1
sim12[:, 1] = r2

sim12_frank = np.empty((frank1.shape[0], 2))
sim12_frank[:, 0] = frank1
sim12_frank[:, 1] = frank2


sns.set_context("paper", font_scale = 1.8)
df12 = pd.DataFrame(sim12, columns=[r"$\epsilon^{u,V}_{\tau\tau}$", r"$\epsilon^{d,V}_{\tau\tau}$"])
sns_plot = sns.jointplot(x=r"$\epsilon^{u,V}_{\tau\tau}$", y=r"$\epsilon^{d,V}_{\tau\tau}$", data=df12,
                         kind="kde")
plt.show()
plt.close()

sns.set_context("paper", font_scale = 1.8)
df_frank_12 = pd.DataFrame(sim12_frank, columns=[r"$\epsilon^{u,V}_{\tau\tau}$", r"$\epsilon^{d,V}_{\tau\tau}$"])
sns_plot = sns.jointplot(x=r"$\epsilon^{u,V}_{\tau\tau}$", y=r"$\epsilon^{d,V}_{\tau\tau}$", data=df_frank_12,
                         kind="kde")
plt.show()
plt.close()

df_data_12 = pd.DataFrame(data_12, columns=[r"$\epsilon^{u,V}_{\tau\tau}$", r"$\epsilon^{d,V}_{\tau\tau}$"])
sns_plot = sns.jointplot(x=r"$\epsilon^{u,V}_{\tau\tau}$", y=r"$\epsilon^{d,V}_{\tau\tau}$", data=df_data_12,
                         kind="kde")
plt.show()
plt.close()