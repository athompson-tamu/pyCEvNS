import numpy as np
from scipy import stats
from scipy.stats import norm
import bisect as bi

import seaborn as sns
import physt


class MNStats:
  def __init__(self, mn_table):
    self.prob = mn_table[:,0]
    self.lnz = mn_table[:,1]
    self.values = mn_table[:,2:8]


  def GetExp(self):
    return np.sum(self.prob.reshape(self.prob.shape[0], 1) * self.values, axis=0)


  def GetVar(self, idx):
    bins, marginal = self.GetMarginal(idx=idx)
    mean = self.GetExp()[idx]
    return np.average((bins - mean) ** 2, weights=marginal)


  def GetMarginal(self, idx):
    nbins = 50
    minx = np.amin(self.values[:, idx])
    maxx = np.amax(self.values[:, idx])
    binw = (maxx - minx) / nbins
    binx = np.linspace(minx + binw / 2, maxx - binw / 2, nbins)
    biny = np.zeros_like(binx)
    for i in range(self.values.shape[0]):
      pos = int((self.values[i, idx] - minx) / binw)
      if pos < nbins:
        biny[pos] += self.prob[i]
      else:
        biny[pos - 1] += self.prob[i]
    return binx, biny



class CombinedStats:
  # Takes the output from three multinest runs and a transformation matrix L
  def __init__(self, mn1, mn2, mn3, L):
    self.e_ = MNStats(mn1)  # (electron)-NSI phenomenological parameter
    self.n_ = MNStats(mn2)  # (nucleus)-NSI phenomenological parameter
    self.o_ = MNStats(mn3)  # (oscillation)-NSI phenomenological parameter
    self.L = L
    self.L_inv = np.linalg.inv(L)
    self.L_inv_2 = np.square(self.L_inv)


  def LTransform(self, x):
    y1 = self.L_inv[0][0] * x[0] + self.L_inv[0][1] * x[1] + self.L_inv[0][2] * x[2]
    y2 = self.L_inv[1][0] * x[0] + self.L_inv[1][1] * x[1] + self.L_inv[1][2] * x[2]
    y3 = self.L_inv[2][0] * x[0] + self.L_inv[2][1] * x[1] + self.L_inv[2][2] * x[2]
    return np.array([y1, y2, y3])


  def MTransform(self, x):
    y1 = self.L_inv_2[0][0] * x[0] + self.L_inv_2[0][1] * x[1] + self.L_inv_2[0][2] * x[2]
    y2 = self.L_inv_2[1][0] * x[0] + self.L_inv_2[1][1] * x[1] + self.L_inv_2[1][2] * x[2]
    y3 = self.L_inv_2[2][0] * x[0] + self.L_inv_2[2][1] * x[1] + self.L_inv_2[2][2] * x[2]
    return np.array([y1, y2, y3])


  def LExp(self):
    e1 = np.sum((self.e_.prob).reshape((self.e_.prob).shape[0], 1) * (self.e_.values), axis=0)
    e2 = np.sum((self.n_.prob).reshape((self.n_.prob).shape[0], 1) * (self.n_.values), axis=0)
    e3 = np.sum((self.o_.prob).reshape((self.o_.prob).shape[0], 1) * (self.o_.values), axis=0)
    return self.LTransform(np.array([e1, e2, e3]))


  def MVar(self, idx):
    var1 = self.e_.GetVar(idx=idx)
    var2 = self.n_.GetVar(idx=idx)
    var3 = self.o_.GetVar(idx=idx)

    # Put in array like [ , , ]
    var_vect = np.array([var1, var2, var3])

    # Apply the M transform to the array.
    return self.MTransform(var_vect), var_vect


  def GetCovMatr(self, idx):
    y_vars, x_vars = self.MVar(idx)
    rho = np.empty((3,3))
    for i in range(0, 3):
      for j in range(0, 3):
        rho[i][j] = np.sum(self.L[i] * self.L[j] * y_vars) / np.sqrt(x_vars[i] * x_vars[j])
    return rho


  def GausCopula(self, R, idx):
    # Take the Cholesky decomp of the covariance matrix R.
    ch = np.linalg.cholesky(R)

    # Simulate independent z1, z2, z3 from N(0,1) and transform.
    z = np.array([norm.rvs(size=50), norm.rvs(size=50), norm.rvs(size=50)])
    w = np.dot(ch.T, z)
    u = np.array([norm.cdf(w[0]), norm.cdf(w[1]), norm.cdf(w[2])])

    # Map the U's back to X's using the inverses of the marginals on the X's.
    r1, f1 = self.e_.GetMarginal(idx)
    r2, f2 = self.n_.GetMarginal(idx)
    r3, f3 = self.o_.GetMarginal(idx)

    x = np.empty_like(u)

    for i in range(0, u.shape[0]):
      left_idx_1 = bi.bisect_left(f1, u[i, 0]) - 1
      left_idx_2 = bi.bisect_left(f2, u[i, 1]) - 1
      left_idx_3 = bi.bisect_left(f3, u[i, 2]) - 1
      f1_inv = r1[left_idx_1]
      f2_inv = r2[left_idx_2]
      f3_inv = r3[left_idx_3]

      x[:,i] = np.array([f1_inv, f2_inv, f3_inv])

    # Now we have joint distribution of X. Use the inv of L to get a distribution on Y.
    y = np.dot(self.L_inv, x)

    return y







def main():
  # Read in posterior distributions.
  print("Reading in files...")
  pheno_1 = np.genfromtxt("multinest_posteriors/all_nsi_vector_solar.txt")
  pheno_2 = np.genfromtxt("multinest_posteriors/all_nsi_xenon_atmos.txt")
  pheno_3 = np.genfromtxt("multinest_posteriors/all_nsi_dune_atmos_mu.txt")

  # Set up constants for Xenon.
  n = 78
  z = 54

  # Define transformation matrix.
  u = np.array([
    [1, 0, 0],
    [0, 1, (2 * n + z) / (2 * z + n)],
    [1, 3, 3]
  ])

  comb_stat = CombinedStats(pheno_1, pheno_2, pheno_3, L=u)
  print(comb_stat.LExp())
  print(comb_stat.MVar(idx=0))

  cov_matr = comb_stat.GetCovMatr(idx=0)
  print(cov_matr)
  copula_ee = comb_stat.GausCopula(R=cov_matr, idx=0)






if __name__ == "__main__":
  main()
