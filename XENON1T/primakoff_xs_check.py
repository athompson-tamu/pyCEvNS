import numpy as np
from numpy import sqrt, pi

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
import matplotlib.ticker as tickr
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import sys
sys.path.append("../")
from pyCEvNS.axion import primakoff_scattering_xs, _screening
from pyCEvNS.constants import *



masses = np.logspace(-7, 0, 1000)

# Benchmark mass models
xs = np.array([primakoff_scattering_xs(0.02, 32, ma, 1) for ma in masses])
screening = np.array([_screening(0.02, ma) for ma in masses])

xs2 = np.array([primakoff_scattering_xs(0.05, 32, ma, 1) for ma in masses])
screening2 = np.array([_screening(0.05, ma) for ma in masses])

xs3 = np.array([primakoff_scattering_xs(0.1, 32, ma, 1) for ma in masses])
screening3 = np.array([_screening(0.1, ma) for ma in masses])

# Plot mass dependence
plt.plot(masses, xs, label="xs 1 MeV")
plt.plot(masses, xs2, ls='dashed', label="xs 5 MeV")
plt.plot(masses, xs3, ls='dotted', label="xs 0.5 MeV")
#plt.xscale('log')
plt.xlim((0, 0.2))
plt.legend()
plt.show()
plt.close()


# Plot energy dep
energies = np.linspace(0.0002,0.01, 1000)
xs = np.array([primakoff_scattering_xs(er, 32, 0.000001, 1) for er in energies])
xs2 = np.array([primakoff_scattering_xs(er, 32, 0.0001, 1) for er in energies])
xs3 = np.array([primakoff_scattering_xs(er, 32, 0.001, 1) for er in energies])
plt.plot(energies, xs, label="xs 1 eV")
plt.plot(energies, xs2, ls='dashed', label="xs 100 eV")
plt.plot(energies, xs3, ls='dotted', label="xs 1 keV")
#plt.xscale('log')
plt.legend()
plt.show()
plt.close()