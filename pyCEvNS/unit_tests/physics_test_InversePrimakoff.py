import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../../")
from pyCEvNS.axion import primakoff_scattering_xs_complete,primakoff_scattering_xs
from pyCEvNS.constants import *


# Reproduce Avignone '88

cu_energy = 1.115
z = 1
r0Ge = 1e-10 / meter_by_mev
r0Xe = 2.18e-10 / meter_by_mev

masses = np.linspace(0,1,1000)
energies = np.logspace(-9,-1,1000)

"""
xs_avignone = [primakoff_scattering_xs_complete(cu_energy, 1, m, z, r0Xe) for m in masses]

plt.plot(masses, xs_avignone, color="red")
plt.xlim((0,1))
plt.show()
plt.close()
"""


xs_3keV = [primakoff_scattering_xs_complete(ea, 1, 0.003, z, r0Xe) for ea in energies]
xs_1keV = [primakoff_scattering_xs_complete(ea, 1, 0.001, z, r0Xe) for ea in energies]
xs_1meV = [primakoff_scattering_xs_complete(ea, 1, 0.00000001, z, r0Xe) for ea in energies]

xs_3keV_old = [primakoff_scattering_xs(ea, z, 0.003, 1) for ea in energies]
xs_1keV_old = [primakoff_scattering_xs(ea, z, 0.001, 1) for ea in energies]
xs_1meV_old = [primakoff_scattering_xs(ea, z, 0.00000001, 1) for ea in energies]

plt.plot(energies, xs_3keV, color="red")
plt.plot(energies, xs_1keV, color="red", ls='dashed')
plt.plot(energies, xs_1meV, color="red", ls='dotted')
plt.plot(energies, xs_3keV_old, color="blue")
plt.plot(energies, xs_1keV_old, color="blue", ls='dashed')
plt.plot(energies, xs_1meV_old, color="blue", ls='dotted')
plt.xlim((0.00000001,0.03))
#plt.yscale('log')
#plt.xscale('log')
plt.show()
plt.close()