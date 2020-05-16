import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from pyCEvNS.axion import primakoff_scattering_xs, IsotropicAxionFromPrimakoff


isoprim = IsotropicAxionFromPrimakoff(photon_rates=[[1,1]], axion_mass=0.1, axion_coupling=2.7e-8)

print(r"testing a + Z $\to \gamma$ + Z")
mass_array = np.linspace(0, 1.2, 100)
xs_array = np.zeros_like(mass_array)
for i, m in enumerate(mass_array):
    xs_array[i] = primakoff_scattering_xs(1.115, 32, ma=m, g=2.7e-8)


mev2_to_barn = 0.00257
plt.plot(mass_array, 1e-24 * xs_array/mev2_to_barn)
plt.show()

