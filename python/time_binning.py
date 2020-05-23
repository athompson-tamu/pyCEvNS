import numpy as np
import matplotlib.pyplot as plt
pe_per_mev = 10000

def get_energy_bins(e_a, e_b):
  return np.arange(e_a, e_b, step=10000 / pe_per_mev)

hi_energy_cut = 100  # mev
lo_energy_cut = 0.0  # mev
energy_edges = get_energy_bins(lo_energy_cut, hi_energy_cut)
energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
timing_edges = np.linspace(0, 0.7, 10)
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
flat_index = 0
for i in range(0, energy_bins.shape[0]):
  for j in range(0, timing_bins.shape[0]):
    n_meas[flat_index, 0] = energy_bins[i] * pe_per_mev
    n_meas[flat_index, 1] = timing_bins[j]
    flat_index += 1

times = np.random.random(100)

time_bin_width = timing_edges[1] - timing_edges[0]
tmin = n_meas[:, 1].min()
tmax = n_meas[:, 1].max()
time_nbins = np.unique(n_meas[:, 1]).shape[0]
t_hist = np.histogram(times, bins=timing_edges, density=True)
plist = t_hist[0]*time_bin_width


print(plist)
print(np.sum(plist))