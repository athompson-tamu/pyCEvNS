import numpy as np


class MNStats:
  def __init__(self, mn_table):
    self.prob = mn_table[:,0]
    self.lnz = mn_table[:,1]
    self.values = mn_table[:,2:]
    self.nbins = 100
    self.ex = self.GetExp()
    self.ex2 = self.GetExp2()
    self.var = self.GetVars()
    self.cdf = np.cumsum(self.prob)


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


  def GetExtrema(self):
    return self.values.max(axis=0), self.values.min(axis=0)
