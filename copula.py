import numpy as np
from scipy import stats
from scipy.stats import norm, multivariate_normal, chi2, t
import bisect as bi
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import physt

from copulae import NormalCopula, StudentCopula, GumbelCopula, ClaytonCopula, FrankCopula

import plotly.graph_objects as go

from statsmodels.stats.weightstats import DescrStatsW
"""
import pynverse as pynv
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector
from rpy2.robjects import r
base = importr('base')
utils = importr('utils')
rcopula = importr('copula')
"""


class SurTawnT1Copula:
  def __init__(self, theta, psi):
    self.theta = theta
    self.psi = psi

  def simulate(self, u, v):
    p = self.psi
    th = self.theta
    def ddv(v2):
      t = np.log(u) / (np.log(u * v2))
      expr = ((1 - t) ** (1/th) + (p * t) ** (1/th))
      exponent = -p + expr ** th
      return (u ** (p - 1)) * v2 * ((u * v2) ** exponent) * p * (expr ** (th-1)) * ((p * t) ** (-1 + 1/th))

    try:
      return 1-float(pynv.inversefunc(ddv, y_values=v, domain=[0, 1], open_domain=[True, True]))
    except:
      pass


# Simulate bivariate pairs empirically.
class EmpricalCopula:
  def __init__(self, datastr, i, j):
    # read in table
    robjects.r('data = read.table(file = "{0}", header=F)'.format(datastr))
    robjects.r('z = pobs(as.matrix(cbind(data[,{0}],data[,{1}])))'.format(i, j))
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
      return v



class MNStats:
  def __init__(self, mn_table):
    self.prob = mn_table[:,0]
    self.lnz = mn_table[:,1]
    self.values = mn_table[:,2:]
    self.nbins = 100
    self.ex = self.GetExp()
    self.ex2 = self.GetExp2()
    self.var = self.GetVars()
    self.corr = self.GetCorrMatr()


  def GetExp(self):
    return np.sum(self.prob.reshape(self.prob.shape[0], 1) * self.values, axis=0)


  def GetExp2(self):
    return np.sum(self.prob.reshape(self.prob.shape[0], 1) * np.square(self.values), axis=0)


  # TODO(me): use 2d marginal instead.
  def GetExpMixed(self, x_i, x_j):
    bins_i, bins_j, marginal2d = self.GetMarginal2d(x_i, x_j)
    product = 0
    for i in range(0, bins_i.shape[0]):
      for j in range(0, bins_j.shape[0]):
        product += bins_i[i] * bins_j[j] * marginal2d[i][j]
    return product


  def GetVars(self):
    vars = np.zeros(self.values.shape[1])
    means = self.GetExp()
    for i in range(0, self.values.shape[1]):
      bins, weights = self.GetMarginal(idx=i)
      vars[i] = np.average(np.square(bins - means[i]), weights=weights)
    return vars


  def GetMarginal(self, idx):
    minx = np.amin(self.values[:, idx])
    maxx = np.amax(self.values[:, idx])
    binw = (maxx - minx) / self.nbins
    binx = np.linspace(minx + binw / 2, maxx - binw / 2, self.nbins)
    biny = np.zeros_like(binx)
    for i in range(self.values.shape[0]):
      pos = int((self.values[i, idx] - minx) / binw)
      if pos < self.nbins:
        biny[pos] += self.prob[i]
      else:
        biny[pos - 1] += self.prob[i]
    return binx, biny


  def GetMarginal2d(self, x_i, x_j):
    minx = np.amin(self.values[:, x_i])
    miny = np.amin(self.values[:, x_j])
    maxx = np.amax(self.values[:, x_i])
    maxy = np.amax(self.values[:, x_j])
    binxw = (maxx - minx) / self.nbins
    binyw = (maxy - miny) / self.nbins
    binx = np.linspace(minx + binxw / 2, maxx - binxw / 2, self.nbins)
    biny = np.linspace(miny + binyw / 2, maxy - binyw / 2, self.nbins)
    xv, yv = np.meshgrid(binx, biny)
    zv = np.zeros_like(xv)
    # be careful that position in x direction is column, position in y direction is row!
    for i in range(self.values.shape[0]):
      posx = int((self.values[i, x_i] - minx) / binxw)
      posy = int((self.values[i, x_j] - miny) / binyw)
      if posx < self.nbins and posy < self.nbins:
        zv[posy, posx] += self.prob[i]
      elif posy < self.nbins:
        zv[posy, posx - 1] += self.prob[i]
      elif posx < self.nbins:
        zv[posy - 1, posx] += self.prob[i]
      else:
        zv[posy - 1, posx - 1] += self.prob[i]
    return binx, biny, zv


  def GetCorrMatr(self):
    corr = np.empty((6, 6))
    for alpha in range(0, 6):
      for beta in range(0, 6):
        exp_mixed = self.GetExpMixed(alpha, beta)
        corr[alpha][beta] = (exp_mixed - self.ex[alpha] * self.ex[beta]) / np.sqrt(self.var[alpha] * self.var[beta])
    return corr


  def GetExtrema(self):
    return self.values.max(axis=0), self.values.min(axis=0)


  # Assuming 6 nsi.
  def GausCopula(self, R, n):
    print(R)
    x1, p1 = self.GetMarginal(0)
    x2, p2 = self.GetMarginal(1)
    x3, p3 = self.GetMarginal(2)
    x4, p4 = self.GetMarginal(3)
    x5, p5 = self.GetMarginal(4)
    x6, p6 = self.GetMarginal(5)

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    cdf3 = np.cumsum(p3)
    cdf4 = np.cumsum(p4)
    cdf5 = np.cumsum(p5)
    cdf6 = np.cumsum(p6)

    ch = np.linalg.cholesky(R)


    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    z = np.array([norm.rvs(size=n), norm.rvs(size=n), norm.rvs(size=n),
                  norm.rvs(size=n), norm.rvs(size=n), norm.rvs(size=n)])

    print("Transforming variates...")
    w = np.dot(ch.T, z)
    u = np.array([norm.cdf(w[0]), norm.cdf(w[1]), norm.cdf(w[2]),
                  norm.cdf(w[3]), norm.cdf(w[4]), norm.cdf(w[5])])

    r1 = np.empty(z.shape[1])
    r2 = np.empty(z.shape[1])
    r3 = np.empty(z.shape[1])
    r4 = np.empty(z.shape[1])
    r5 = np.empty(z.shape[1])
    r6 = np.empty(z.shape[1])

    for i in range(0, u.shape[1]):
      r1[i] = np.interp(u[0, i], cdf1, x1)
      r2[i] = np.interp(u[1, i], cdf2, x2)
      r3[i] = np.interp(u[2, i], cdf3, x3)
      r4[i] = np.interp(u[3, i], cdf4, x4)
      r5[i] = np.interp(u[4, i], cdf5, x5)
      r6[i] = np.interp(u[5, i], cdf6, x6)

    return r1, r2, r3, r4, r5, r6

  def GumbCopula(self, theta, n):
    x1, p1 = self.GetMarginal(0)
    x2, p2 = self.GetMarginal(1)
    #x3, p3 = self.GetMarginal(2)
    #x4, p4 = self.GetMarginal(3)
    #x5, p5 = self.GetMarginal(4)
    #x6, p6 = self.GetMarginal(5)

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    #cdf3 = np.cumsum(p3)
    #cdf4 = np.cumsum(p4)
    #cdf5 = np.cumsum(p5)
    #cdf6 = np.cumsum(p6)

    cop = GumbelCopula(theta=theta, dim=2)
    u = cop.random(n)

    r1 = np.empty(n)
    r2 = np.empty(n)
    #r3 = np.empty(n)
    #r4 = np.empty(n)
    #r5 = np.empty(n)
    #r6 = np.empty(n)

    for i in range(0, u.shape[0]):
      r1[i] = np.interp(u[i, 0], cdf1, x1)
      r2[i] = np.interp(u[i, 1], cdf2, x2)
      #r3[i] = np.interp(u[i, 2], cdf3, x3)
      #r4[i] = np.interp(u[i, 3], cdf4, x4)
      #r5[i] = np.interp(u[i, 4], cdf5, x5)
      #r6[i] = np.interp(u[i, 5], cdf6, x6)

    return r1, r2 #, r3, r4, r5, r6


  def ArchCopula(self, theta, n, family):
    x1, p1 = self.GetMarginal(1)
    x2, p2 = self.GetMarginal(7)
    #x3, p3 = self.GetMarginal(2)
    #x4, p4 = self.GetMarginal(3)
    #x5, p5 = self.GetMarginal(4)
    #x6, p6 = self.GetMarginal(5)

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    #cdf3 = np.cumsum(p3)
    #cdf4 = np.cumsum(p4)
    #cdf5 = np.cumsum(p5)
    #cdf6 = np.cumsum(p6)

    cop = ClaytonCopula(theta=theta, dim=2)
    if family == "Clayton":
      cop = ClaytonCopula(theta=theta, dim=2)
    if family == "Gumbel":
      cop = GumbelCopula(theta=theta, dim=2)
    if family == "Frank":
      cop = FrankCopula(theta=theta, dim=2)
    u = cop.random(n)

    r1 = np.empty(n)
    r2 = np.empty(n)
    #r3 = np.empty(n)
    #r4 = np.empty(n)
    #r5 = np.empty(n)
    #r6 = np.empty(n)

    for i in range(0, u.shape[0]):
      r1[i] = np.interp(u[i, 0], cdf1, x1)
      r2[i] = np.interp(u[i, 1], cdf2, x2)
      #r3[i] = np.interp(u[i, 2], cdf3, x3)
      #r4[i] = np.interp(u[i, 3], cdf4, x4)
      #r5[i] = np.interp(u[i, 4], cdf5, x5)
      #r6[i] = np.interp(u[i, 5], cdf6, x6)

    return r1, r2 #, r3, r4, r5, r6


  def tCopula(self, R, nu, n):
    print(R)
    x1, p1 = self.GetMarginal(0)
    x2, p2 = self.GetMarginal(1)
    x3, p3 = self.GetMarginal(2)
    x4, p4 = self.GetMarginal(3)
    x5, p5 = self.GetMarginal(4)
    x6, p6 = self.GetMarginal(5)

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    cdf3 = np.cumsum(p3)
    cdf4 = np.cumsum(p4)
    cdf5 = np.cumsum(p5)
    cdf6 = np.cumsum(p6)

    multinorm = multivariate_normal(cov=R)
    chisquare = chi2(nu)


    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    z = multinorm.rvs(size=n)
    xi = chisquare.rvs(size=n)
    u = np.empty_like(z)
    u[:, 0] = t.cdf(z[:, 0] / (np.sqrt(xi / 2)), nu)
    u[:, 1] = t.cdf(z[:, 1] / (np.sqrt(xi / 2)), nu)
    u[:, 2] = t.cdf(z[:, 2] / (np.sqrt(xi / 2)), nu)
    u[:, 3] = t.cdf(z[:, 3] / (np.sqrt(xi / 2)), nu)
    u[:, 4] = t.cdf(z[:, 4] / (np.sqrt(xi / 2)), nu)
    u[:, 5] = t.cdf(z[:, 5] / (np.sqrt(xi / 2)), nu)


    r1 = np.empty(z.shape[0])
    r2 = np.empty(z.shape[0])
    r3 = np.empty(z.shape[0])
    r4 = np.empty(z.shape[0])
    r5 = np.empty(z.shape[0])
    r6 = np.empty(z.shape[0])

    for i in range(0, u.shape[0]):
      r1[i] = np.interp(u[i, 0], cdf1, x1)
      r2[i] = np.interp(u[i, 1], cdf2, x2)
      r3[i] = np.interp(u[i, 2], cdf3, x3)
      r4[i] = np.interp(u[i, 3], cdf4, x4)
      r5[i] = np.interp(u[i, 4], cdf5, x5)
      r6[i] = np.interp(u[i, 5], cdf6, x6)

    return r1, r2, r3, r4, r5, r6


  def tMixCopula(self, R, nu, n):
    print(R)
    Rp = -R + 2*np.identity(3)
    print(Rp)
    x1, p1 = self.GetMarginal(0)
    x2, p2 = self.GetMarginal(1)
    x3, p3 = self.GetMarginal(2)

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    cdf3 = np.cumsum(p3)

    multinorm = multivariate_normal(cov=R)
    multinormp = multivariate_normal(cov=Rp)
    chisquare = chi2(nu)


    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    z = multinorm.rvs(size=n)
    zp = multinormp.rvs(size=n)
    xi = chisquare.rvs(size=n)
    xip = chisquare.rvs(size=n)
    u = np.empty_like(z)
    u[:, 0] = 0.5 * t.cdf(z[:, 0] / (np.sqrt(xi / 2)), nu) \
              + 0.5 * t.cdf(zp[:, 0] / (np.sqrt(xip / 2)), nu)
    u[:, 1] = 0.5 * t.cdf(z[:, 1] / (np.sqrt(xi / 2)), nu) \
              + 0.5 * t.cdf(zp[:, 1] / (np.sqrt(xip / 2)), nu)
    u[:, 2] = 0.5 * t.cdf(z[:, 2] / (np.sqrt(xi / 2)), nu) \
              + t.cdf(zp[:, 2] / (np.sqrt(xip / 2)), nu)


    r1 = np.empty(z.shape[0])
    r2 = np.empty(z.shape[0])
    r3 = np.empty(z.shape[0])

    for i in range(0, u.shape[0]):
      r1[i] = np.interp(u[i, 0], cdf1, x1)
      r2[i] = np.interp(u[i, 1], cdf2, x2)
      r3[i] = np.interp(u[i, 2], cdf3, x3)

    return r1, r2, r3


  def EllipticalCopula(self, R, nu, n):
    print(R)
    x1, p1 = self.GetMarginal(0)
    x2, p2 = self.GetMarginal(1)
    x3, p3 = self.GetMarginal(2)
    x4, p4 = self.GetMarginal(3)
    x5, p5 = self.GetMarginal(4)
    x6, p6 = self.GetMarginal(5)
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    cdf3 = np.cumsum(p3)
    cdf4 = np.cumsum(p4)
    cdf5 = np.cumsum(p5)
    cdf6 = np.cumsum(p6)

    ch = np.linalg.cholesky(R)

    chisquare = chi2(nu)

    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    print("Transforming variates...")
    z = np.array([norm.rvs(size=n), norm.rvs(size=n), norm.rvs(size=n),
                  norm.rvs(size=n), norm.rvs(size=n), norm.rvs(size=n)])
    modz = np.sqrt(z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2 + z[4]**2 + z[5]**2)
    s = z / modz
    xi = chisquare.rvs(size=n)
    x = np.sqrt(xi) * np.dot(ch.T, s)

    u = np.array([norm.cdf(x[0]), norm.cdf(x[1]), norm.cdf(x[2]),
                  norm.cdf(x[3]), norm.cdf(x[4]), norm.cdf(x[5])])

    r1 = np.empty(z.shape[1])
    r2 = np.empty(z.shape[1])
    r3 = np.empty(z.shape[1])
    r4 = np.empty(z.shape[1])
    r5 = np.empty(z.shape[1])
    r6 = np.empty(z.shape[1])

    for i in range(0, u.shape[1]):
      r1[i] = np.interp(u[0, i], cdf1, x1)
      r2[i] = np.interp(u[1, i], cdf2, x2)
      r3[i] = np.interp(u[2, i], cdf3, x3)
      r4[i] = np.interp(u[3, i], cdf4, x4)
      r5[i] = np.interp(u[4, i], cdf5, x5)
      r6[i] = np.interp(u[5, i], cdf6, x6)

    return r1, r2, r3, r4, r5, r6



class CombinedStats:
  # Takes the output from three multinest runs and a transformation matrix L
  def __init__(self, mn1, mn2, mn3, L, samples):
    self.e_ = MNStats(mn1)  # (electron)-NSI phenomenological parameter
    self.n_ = MNStats(mn2)  # (nucleus)-NSI phenomenological parameter
    self.o_ = MNStats(mn3)  # (oscillation)-NSI phenomenological parameter
    self.L = L
    self.L_inv = np.linalg.inv(L)
    self.L_inv_2 = np.square(self.L_inv)
    self.M = np.square(L)
    self.M_inv = np.linalg.inv(self.M)
    self.do_plots = False
    self.nbins = samples
    self.nsi = {0: 'ee', 1: 'mm', 2: 'tt', 3: 'em', 4: 'et', 5: 'mt'}
    self.f = {0: 'e', 1: 'u', 2: 'd'}


  def TogglePlotting(self, toggle):
    self.do_plots = toggle


  def LTransform(self, x):
    y1 = self.L_inv[0][0] * x[0] + self.L_inv[0][1] * x[1] + self.L_inv[0][2] * x[2]
    y2 = self.L_inv[1][0] * x[0] + self.L_inv[1][1] * x[1] + self.L_inv[1][2] * x[2]
    y3 = self.L_inv[2][0] * x[0] + self.L_inv[2][1] * x[1] + self.L_inv[2][2] * x[2]
    return np.array([y1, y2, y3])


  def MTransform(self, x):
    y1 = self.M_inv[0][0] * x[0] + self.M_inv[0][1] * x[1] + self.M_inv[0][2] * x[2]
    y2 = self.M_inv[1][0] * x[0] + self.M_inv[1][1] * x[1] + self.M_inv[1][2] * x[2]
    y3 = self.M_inv[2][0] * x[0] + self.M_inv[2][1] * x[1] + self.M_inv[2][2] * x[2]
    return np.array([y1, y2, y3])


  def GetTransformedExtrema(self):
    e_max, e_min = self.e_.GetExtrema()
    n_max, n_min = self.n_.GetExtrema()
    o_max, o_min = self.o_.GetExtrema()
    return self.LTransform(np.array([e_max, n_max, o_max])),\
           self.LTransform(np.array([e_min, n_min, o_min]))


  def MVar(self, idx):
    var1 = self.e_.GetVars[idx]
    var2 = self.n_.GetVars[idx]
    var3 = self.o_.GetVars[idx]
    var_vect = np.array([var1, var2, var3])
    return self.MTransform(var_vect), var_vect


  def MVars(self):
    var1 = self.e_.GetVars()
    var2 = self.n_.GetVars()
    var3 = self.o_.GetVars()
    var_vect = np.array([var1, var2, var3])
    return self.MTransform(var_vect), var_vect



  def GetCorrMatr(self, idx):
    y_vars, x_vars = self.MVar(idx)
    rho = np.empty((3,3))
    for i in range(0, 3):
      for j in range(0, 3):
        rho[i][j] = np.sum(self.L[i] * self.L[j] * y_vars) / np.sqrt(x_vars[i] * x_vars[j])
    return rho


  def GetHeuristicCorrMatr(self):
    x_vars = self.MVars()[1]
    rho = np.zeros((18,18))
    for alpha in range(0, 6):
      for beta in range(0, 6):
        e_mixed = self.e_.GetExpMixed(alpha, beta)
        n_mixed = self.n_.GetExpMixed(alpha, beta)
        cov_13 = 3 * self.e_.ex[alpha] * self.n_.ex[beta] + e_mixed - self.e_.ex[alpha] * self.o_.ex[beta]
        cov_23 = self.e_.ex[alpha] * self.n_.ex[beta] + 3 * n_mixed - self.n_.ex[alpha] * self.o_.ex[beta]
        corr_13 = cov_13 / np.sqrt(x_vars[0][alpha] * x_vars[2][beta])
        corr_23 = cov_23 / np.sqrt(x_vars[1][alpha] * x_vars[2][beta])
        rho[3*alpha][3*beta] = 1
        rho[3*alpha + 1][3*beta + 1] = 1
        rho[3*alpha + 2][3*beta + 2] = 1
        rho[3*alpha][3*beta+1] = 0
        rho[3*alpha+1][beta] = 0
        rho[3*alpha][3*beta + 2] = corr_13
        rho[3 * alpha + 2][3 * beta] = corr_13
        rho[3*alpha+1][3*beta+2] = corr_23
        rho[3*alpha+2][3*beta+1] = corr_23

    # Apply shrinking method to get pseudo correlation matrix
    while np.any(np.linalg.eigvals(rho) < 0):
      lambd = 0.99
      rho = (rho * lambd) + (1 - lambd) * np.identity(18)

    return rho


  def GetSingleHeuristicCorrMatr(self, alpha):  # alpha = 0,...,5 for the 6 nsi.
    x_vars = self.MVars()[1]
    cov_13 = 3 * self.e_.ex[alpha] * self.n_.ex[alpha] + self.e_.ex2[alpha] - self.e_.ex[alpha] * self.o_.ex[alpha]
    cov_23 = self.e_.ex[alpha] * self.n_.ex[alpha] + 3 * self.n_.ex2[alpha] - self.n_.ex[alpha] * self.o_.ex[alpha]
    corr_13 = cov_13 / np.sqrt(x_vars[0][alpha] * x_vars[2][beta])
    corr_23 = cov_23 / np.sqrt(x_vars[1][alpha] * x_vars[2][beta])
    rho = np.array([[1, 0, corr_13],
                    [0, 1, corr_23],
                    [corr_13, corr_23, 1]])

    # Apply shrinking method to get pseudo correlation matrix
    while np.any(np.linalg.eigvals(rho) < 0):
      lambd = 0.99
      rho = (rho * lambd) + (1 - lambd) * np.identity(3)

    return rho


  def GetJointCorrMatr(self, alpha, beta):
    x_vars = self.MVars()[1]

    e_mixed = self.e_.GetExpMixed(alpha, beta)
    n_mixed = self.n_.GetExpMixed(alpha, beta)
    o_mixed = self.o_.GetExpMixed(alpha, beta)
    corr_11 = (e_mixed - self.e_.ex[alpha] * self.e_.ex[beta]) / np.sqrt(x_vars[0][alpha] * x_vars[0][beta])
    corr_22 = (n_mixed - self.n_.ex[alpha] * self.n_.ex[beta]) / np.sqrt(x_vars[1][alpha] * x_vars[1][beta])
    corr_33 = (o_mixed - self.o_.ex[alpha] * self.o_.ex[beta]) / np.sqrt(x_vars[2][alpha] * x_vars[2][beta])
    #cov_13 = 3 * self.e_.ex[alpha] * self.n_.ex[beta] + e_mixed - self.e_.ex[alpha] * self.o_.ex[beta]
    #cov_31 = 3 * self.e_.ex[beta] * self.n_.ex[alpha] + e_mixed - self.e_.ex[beta] * self.o_.ex[alpha]
    #cov_23 = self.e_.ex[alpha] * self.n_.ex[beta] + 3 * n_mixed - self.n_.ex[alpha] * self.o_.ex[beta]
    #cov_32 = self.e_.ex[beta] * self.n_.ex[alpha] + 3 * n_mixed - self.n_.ex[beta] * self.o_.ex[alpha]
    #corr_13 = cov_13 / np.sqrt(x_vars[0][alpha] * x_vars[2][beta])
    #corr_23 = cov_23 / np.sqrt(x_vars[1][alpha] * x_vars[2][beta])
    #corr_31 = cov_31 / np.sqrt(x_vars[0][beta] * x_vars[2][alpha])
    #corr_32 = cov_32 / np.sqrt(x_vars[1][beta] * x_vars[2][alpha])
    return np.array([[corr_11, 0, 0.1 * (alpha == beta)],
                     [0, corr_22, 0.7 * (alpha == beta)],
                     [0.1 * (alpha == beta), 0.7 * (alpha == beta), corr_33]])


  def GetDoubleHeuristicCorrMatr(self, alpha, beta):
    block_rho = np.block([[self.GetJointCorrMatr(alpha, alpha), self.GetJointCorrMatr(alpha, beta)],
                          [self.GetJointCorrMatr(beta, alpha), self.GetJointCorrMatr(beta, beta)]])
    # Apply shrinking method to get pseudo correlation matrix
    while np.any(np.linalg.eigvals(block_rho) < 0):
      scale = 0.99
      block_rho = (block_rho * scale) + (1 - scale) * np.identity(6)

    print(np.round(block_rho, 3))

    return block_rho



  def CorrMatr2d(self, alpha, beta):
    x_vars = self.MVars()[1]
    o_mixed = self.o_.GetExpMixed(alpha, beta)
    corr_12 = (o_mixed - self.o_.ex[alpha] * self.o_.ex[beta]) / np.sqrt(x_vars[2][alpha] * x_vars[2][beta])
    return np.array([[1, corr_12],
                     [corr_12, 1]])


  def CorrMatr2d_XeNuc(self, alpha, beta):
    x_vars = self.MVars()[1]
    n_mixed = self.n_.GetExpMixed(alpha, beta)
    corr_12 = (n_mixed - self.n_.ex[alpha] * self.n_.ex[beta]) / np.sqrt(x_vars[1][alpha] * x_vars[1][beta])
    return np.array([[1, corr_12],
                     [corr_12, 1]])



  def Gaus2DCopula(self, marginal, R, alpha, beta):
    print(R)
    x1, p1 = marginal.GetMarginal(alpha)
    x2, p2 = marginal.GetMarginal(beta)
    ch = np.linalg.cholesky(R)

    mv_norm = norm()

    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    z = np.array([mv_norm.rvs(size=self.nbins), mv_norm.rvs(size=self.nbins)])

    print("Transforming variates...")
    w = np.dot(ch.T, z)
    u = np.array([mv_norm.cdf(w[0]), mv_norm.cdf(w[1])])

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    r1 = np.empty(z.shape[1])
    r2 = np.empty(z.shape[1])

    for i in range(0, u.shape[1]):
      r1[i] = np.interp(u[0, i], cdf1, x1)
      r2[i] = np.interp(u[1, i], cdf2, x2)

    return r1, r2


  def t_2DCopula(self, marginal, R, alpha, beta):
    print(R)
    x1, p1 = marginal.GetMarginal(alpha)
    x2, p2 = marginal.GetMarginal(beta)

    multinorm = multivariate_normal(cov=R)
    chisquare = chi2(4)


    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    z = multinorm.rvs(size=self.nbins)
    xi = chisquare.rvs(size=self.nbins)
    u = np.empty_like(z)
    u[:,0] = t.cdf(z[:,0] / (np.sqrt(xi / 2)), 2)
    u[:,1] = t.cdf(z[:,1] / (np.sqrt(xi / 2)), 2)

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    r1 = np.empty(z.shape[0])
    r2 = np.empty(z.shape[0])

    for i in range(0, u.shape[0]):
      r1[i] = np.interp(u[i, 0], cdf1, x1)
      r2[i] = np.interp(u[i, 1], cdf2, x2)

    return r1, r2


  def EllipticalCopula(self, mn_table, R, alpha, beta):
    print(R)
    x1, p1 = mn_table.GetMarginal(alpha)
    x2, p2 = mn_table.GetMarginal(beta)
    ch = np.linalg.cholesky(R)

    n = norm()
    chisquare = chi2(5)

    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    print("Transforming variates...")
    z = np.array([n.rvs(size=self.nbins), n.rvs(size=self.nbins)])
    modz = np.sqrt(z[0]**2 + z[1]**2)
    s = z / modz
    xi = chisquare.rvs(size=self.nbins)
    x = np.sqrt(xi) * np.dot(ch.T, s)

    u = np.array([n.cdf(x[0]), n.cdf(x[1])])

    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    r1 = np.empty(z.shape[1])
    r2 = np.empty(z.shape[1])

    for i in range(0, u.shape[1]):
      r1[i] = np.interp(u[0, i], cdf1, x1)
      r2[i] = np.interp(u[1, i], cdf2, x2)

    return r1, r2




  def GausCopula(self, R, alpha, beta):
    print("Generating N_6 (0,1)...")

    # Take the Cholesky decomp of the covariance matrix R.
    np.savetxt("corr_matrix.txt", R, delimiter=",")
    print("using covariance matrix = \n", R)
    ch = np.linalg.cholesky(R)

    id = np.identity(R.shape[0])
    mean = np.zeros(R.shape[0])
    mv_norm = multivariate_normal(mean=mean, cov=id)

    # Simulate independent z1, z2, z3 from N_6(0,1) and transform.
    z = np.array([mv_norm.rvs(size=self.nbins), mv_norm.rvs(size=self.nbins), mv_norm.rvs(size=self.nbins),
                  mv_norm.rvs(size=self.nbins), mv_norm.rvs(size=self.nbins), mv_norm.rvs(size=self.nbins)])

    print("Transforming variates...")
    w = np.tensordot(ch.T, z, axes=1)
    u = np.array([mv_norm.cdf(w[0]), mv_norm.cdf(w[1]), mv_norm.cdf(w[2]),
                  mv_norm.cdf(w[3]), mv_norm.cdf(w[4]), mv_norm.cdf(w[5])])

    # Map the U's back to X's using the inverses of F(X)
    # TODO: get marginals instead so we have a separate CDF for each variable.
    cdf1 = np.cumsum(self.e_.prob)
    cdf2 = np.cumsum(self.n_.prob)
    cdf3 = np.cumsum(self.o_.prob)
    ra1 = self.e_.values[:, alpha]
    ra2 = self.n_.values[:, alpha]
    ra3 = self.o_.values[:, alpha]
    rb1 = self.e_.values[:, beta]
    rb2 = self.n_.values[:, beta]
    rb3 = self.o_.values[:, beta]

    r_a = np.empty((3, z.shape[1]))
    r_b = np.empty((3, z.shape[1]))

    print("Getting inverses...")
    for i in range(0, u.shape[1]):
      f1_inv = ra1[bi.bisect_left(cdf1, u[0, i])]
      f2_inv = ra2[bi.bisect_left(cdf2, u[1, i])]
      f3_inv = ra3[bi.bisect_left(cdf3, u[2, i])]
      g1_inv = rb1[bi.bisect_left(cdf1, u[3, i])]
      g2_inv = rb2[bi.bisect_left(cdf2, u[4, i])]
      g3_inv = rb3[bi.bisect_left(cdf3, u[5, i])]

      r_a[:,i] = np.array([f1_inv, f2_inv, f3_inv])
      r_b[:,i] = np.array([g1_inv, g2_inv, g3_inv])

    # Now we have joint distribution of X. Use the inv of L to get a distribution on Y.
    y_a = np.tensordot(self.L_inv, r_a, axes=1)
    y_b = np.tensordot(self.L_inv, r_b, axes=1)
    print("Generating plots...")

    if self.do_plots == True:
      for p in range(0,6):
        #binned_cdf1 = np.histogram(cdf1, bins=self.nbins, range=(0,1))
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15,4))
        ax1.hist(z[0][:,p], bins=100, histtype='step')
        ax1.hist(z[1][:,p], bins=100, histtype='step')
        ax1.hist(z[2][:,p], bins=100, histtype='step')
        ax1.set_xlabel("z")
        ax2.hist(w[0][:,p], bins=100, histtype='step')
        ax2.hist(w[1][:,p], bins=100, histtype='step')
        ax2.hist(w[2][:,p], bins=100, histtype='step')
        ax2.set_xlabel("w")
        ax3.plot(u[0])
        ax3.plot(u[1])
        ax3.plot(u[2])
        ax3.set_xlabel("u")
        ax4.plot(cdf1)
        ax4.plot(cdf2)
        ax4.plot(cdf3)
        ax4.set_xlabel("cdf")
        eps_e_str = "r'$\epsilon^{e,V}_{" + self.nsi[p] + "}$'"
        eps_u_str = "r'$\epsilon^{u,V}_{" + self.nsi[p] + "}$'"
        eps_d_str = "r'$\epsilon^{d,V}_{" + self.nsi[p] + "}$'"
        ax5.hist(y_a[0], bins=100, histtype='step', label=eps_e_str)
        ax5.hist(y_a[1], bins=100, histtype='step', label=eps_u_str)
        ax5.hist(y_a[2], bins=100, histtype='step', label=eps_d_str)
        ax5.legend()
        ax5.set_xlabel("y")
        plt.tight_layout()
        png_str = "png/copula/" + self.nsi[p] + ".png"
        plt.savefig(png_str)
        plt.clf()
    return y_a, y_b



def main():
  # Read in posterior distributions.
  print("Reading in files...")
  post_electron = np.genfromtxt("multinest_posteriors/all_nsi_vector_solar.txt")
  post_nucleus = np.genfromtxt("multinest_posteriors/all_nsi_xenon_atmos.txt")
  post_oscillation = np.genfromtxt("multinest_posteriors/all_nsi_dune_atmos_mu.txt")
  post_coh = np.genfromtxt("multinest_posteriors/coherent_ud_.txt")

  # Save some marginals.
  xee = MNStats(post_coh)
  uee_x, uee_y = xee.GetMarginal(0)
  umm_x, umm_y = xee.GetMarginal(1)
  utt_x, utt_y = xee.GetMarginal(2)
  uem_x, uem_y = xee.GetMarginal(3)
  uet_x, uet_y = xee.GetMarginal(4)
  umt_x, umt_y = xee.GetMarginal(5)
  dee_x, dee_y = xee.GetMarginal(6)
  dmm_x, dmm_y = xee.GetMarginal(7)
  dtt_x, dtt_y = xee.GetMarginal(8)
  dem_x, dem_y = xee.GetMarginal(9)
  det_x, det_y = xee.GetMarginal(10)
  dmt_x, dmt_y = xee.GetMarginal(11)
  mout = np.empty((uee_x.shape[0], 12))
  mout[:, 0] = uee_x
  mout[:, 1] = umm_x
  mout[:, 2] = utt_x
  mout[:, 3] = uem_x
  mout[:, 4] = uet_x
  mout[:, 5] = umt_x
  mout[:, 6] = dee_x
  mout[:, 7] = dmm_x
  mout[:, 8] = dtt_x
  mout[:, 9] = dem_x
  mout[:, 10] = det_x
  mout[:, 11] = dmt_x
  np.savetxt("coh_values.txt", mout)
  print("saved values")
  mout[:, 0] = uee_y
  mout[:, 1] = umm_y
  mout[:, 2] = utt_y
  mout[:, 3] = uem_y
  mout[:, 4] = uet_y
  mout[:, 5] = umt_y
  mout[:, 6] = dee_y
  mout[:, 7] = dmm_y
  mout[:, 8] = dtt_y
  mout[:, 9] = dem_y
  mout[:, 10] = det_y
  mout[:, 11] = dmt_y
  np.savetxt("coh_marginals.txt", mout)

  # Set up constants for Xenon.
  n = 78
  z = 54

  # Define transformation matrix.
  u = np.array([
    [1, 0, 0],
    [0, 1, (2 * n + z) / (2 * z + n)],
    [1, 3, 3]
  ])


  # COPULA vs. MULTINEST
  print("Setting up...")
  comb = CombinedStats(post_electron, post_nucleus, post_oscillation, L=u, samples=1000)
  R01_dune = comb.CorrMatr2d(0, 1)
  R02_dune = comb.CorrMatr2d(0, 2)
  R13_dune = comb.CorrMatr2d(1, 3)
  R03_dune = comb.CorrMatr2d(0, 3)
  R34_dune = comb.CorrMatr2d(3, 4)
  R35_dune = comb.CorrMatr2d(3, 5)
  R45_dune = comb.CorrMatr2d(4, 5)
  R56_dune = comb.CorrMatr2d(5, 6)
  R01_xe = comb.CorrMatr2d_XeNuc(0, 1)
  R02_xe = comb.CorrMatr2d_XeNuc(0, 2)
  R13_xe = comb.CorrMatr2d_XeNuc(1, 3)
  R03_xe = comb.CorrMatr2d_XeNuc(0, 3)
  R34_xe = comb.CorrMatr2d_XeNuc(3, 4)
  R35_xe = comb.CorrMatr2d_XeNuc(3, 5)
  R45_xe = comb.CorrMatr2d_XeNuc(4, 5)
  R56_xe = comb.CorrMatr2d_XeNuc(5, 6)

  print("(4,5)..")
  r1, r2 = comb.t_2DCopula(comb.n_, R45_xe, 4, 5)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{e\tau}$", r"$\epsilon^{Xe-N}_{\mu\tau}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{e\tau}$", y=r"$\epsilon^{Xe-N}_{\mu\tau}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_et_mt.png")

  print("(0,1)..")
  r1, r2 = comb.EllipticalCopula(comb.n_, R01_xe, 0, 1)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{ee}$", r"$\epsilon^{Xe-N}_{\mu\mu}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{ee}$", y=r"$\epsilon^{Xe-N}_{\mu\mu}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_ee_mm.png")

  print("(0,2)..")
  r1, r2 = comb.EllipticalCopula(comb.n_, R02_xe, 0, 2)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{ee}$", r"$\epsilon^{Xe-N}_{\tau\tau}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{ee}$", y=r"$\epsilon^{Xe-N}_{\tau\tau}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_ee_tt.png")

  print("(1,3)..")
  r1, r2 = comb.t_2DCopula(comb.n_, R13_xe, 1, 3)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{\mu\mu}$", r"$\epsilon^{Xe-N}_{e\mu}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{\mu\mu}$", y=r"$\epsilon^{Xe-N}_{e\mu}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_mm_em.png")

  print("(0,3)..")
  r1, r2 = comb.t_2DCopula(comb.n_, R03_xe, 0, 3)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{ee}$", r"$\epsilon^{Xe-N}_{e\mu}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{ee}$", y=r"$\epsilon^{Xe-N}_{e\mu}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_ee_em.png")

  print("(3,4)..")
  r1, r2 = comb.t_2DCopula(comb.n_, R34_xe, 3, 4)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{e\mu}$", r"$\epsilon^{Xe-N}_{e\tau}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{e\mu}$", y=r"$\epsilon^{Xe-N}_{e\tau}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_em_et.png")

  print("(3,5)..")
  r1, r2 = comb.t_2DCopula(comb.n_, R35_xe, 3, 5)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{e\mu}$", r"$\epsilon^{Xe-N}_{\mu\tau}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{e\mu}$", y=r"$\epsilon^{Xe-N}_{\mu\tau}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t-copula_em_mt.png")

  print("(5,6)..")
  r1, r2 = comb.t_2DCopula(comb.n_, R56_xe, 5, 6)
  copula_data = np.empty((r1.shape[0], 2))
  copula_data[:, 0] = r1
  copula_data[:, 1] = r2
  df = pd.DataFrame(copula_data, columns=[r"$\epsilon^{Xe-N}_{\mu\tau}$", r"$\delta_{CP}$"])
  sns_plot = sns.jointplot(x=r"$\epsilon^{Xe-N}_{\mu\tau}$", y=r"$\delta_{CP}$", data=df, kind="kde")
  sns_plot.savefig("png/copula/Xe-N_t_copula_mt_dcp.png")


if __name__ == "__main__":
  main()