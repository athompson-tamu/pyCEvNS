from pyCEvNS.boltzmann import *
import matplotlib.pyplot as plt
import numpy as np


solver, y01 = boltzmann(0.01, 75, alphad=0.5, mf=me, mchi_ratio=3, xinit=1, xfin=10000)


