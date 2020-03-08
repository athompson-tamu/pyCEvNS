import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

limits_0p1dru = np.genfromtxt("limits/electron/miner_csi_0p1dru.txt")
limits_1dru = np.genfromtxt("limits/electron/miner_csi_1dru.txt")
beamee = np.genfromtxt('data/beam_ee.txt', delimiter=',')
barbaree = np.genfromtxt('data/babar_ee.txt', delimiter=',')
elderee = np.genfromtxt('data/edelweis.txt', delimiter=',')

me = 0.511

m_array_0p1dru = limits_0p1dru[:,0]
g_array_0p1dru = limits_0p1dru[:,1]
m_array_1dru = limits_0p1dru[:,0]
g_array_1dru = limits_0p1dru[:,1]


# Draw limits (smoothed)
spline_0p1dru = gaussian_filter1d(g_array_0p1dru, sigma=0.2, mode='nearest')
spline_1dru = gaussian_filter1d(g_array_1dru, sigma=0.01, mode='nearest')
fine_mass_array = np.logspace(1, 7, 100)

fig, ax = plt.subplots()
ax.plot(m_array_0p1dru, spline_0p1dru, color='red', ls="dashed", label='MINER Ge (0.1 DRU)')
ax.plot(m_array_1dru, spline_1dru, color='red', label='MINER Ge (1 DRU)')
ax.fill(beamee[:, 0] * 1e6, beamee[:, 1] / 1e3 * (me), label='Beam Dump', alpha=0.5)
ax.fill(barbaree[:, 0] * 1e6, barbaree[:, 1] / 1e3 * (me), label='BaBar', alpha=0.5)
ax.fill(np.hstack((elderee[:, 0], np.min(elderee[:, 0]))) * 1e6,
        np.hstack((elderee[:, 1], np.max(elderee[:, 1]))) / 1e3 * (me), label='Edelweiss', alpha=0.5)
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$m_a$ [eV]')
ax.set_ylabel(r'$g_{ae}$ [GeV$^{-1}$]')
ax.set_xlim(10, 1e7)
ax.set_ylim(5e-6, 1)
fig.tight_layout()
fig.savefig('plots/miner_limits/MINER_limits_electron_csi_DRU_smooth.pdf')
fig.savefig('plots/miner_limits/MINER_limits_electron_csi_DRU_smooth.png')