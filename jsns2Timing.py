from scipy import signal
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline



prompt_pdf = np.genfromtxt('data/jsns/pion_kaon_neutrino_timing.txt', delimiter=',')
delayed_pdf = np.genfromtxt('data/jsns/mu_neutrino_timing.txt', delimiter=',')

def prompt_time(t):
    return np.interp(1000*t, prompt_pdf[:,0], prompt_pdf[:,1])

def delayed_time(t):
  return np.interp(1000 * t, delayed_pdf[:, 0], delayed_pdf[:, 1])

integral_delayed = quad(delayed_time, 0, 2)[0]
integral_prompt = quad(prompt_time, 0, 2)[0]

def prompt_prob(ta, tb):
  return quad(prompt_time, ta, tb)[0] / integral_prompt

def delayed_prob(ta, tb):
  return quad(delayed_time, ta, tb)[0] / integral_delayed



timing_edges = np.linspace(0,2,200)
timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2

times = np.linspace(0,2,1000)

binned_delayed = np.empty(timing_bins.shape[0])
binned_prompt = np.empty(timing_bins.shape[0])
for i in range(0, binned_prompt.shape[0]):
    binned_delayed[i] = delayed_prob(timing_edges[i], timing_edges[i+1])
    binned_prompt[i] = prompt_prob(timing_edges[i], timing_edges[i+1])


plt.plot(timing_bins, binned_delayed, ls='steps-mid', color='b', label='delayed')
plt.plot(timing_bins, binned_prompt, ls='steps-mid', color='r', label='prompt')

plt.plot(times, delayed_time(times), color='k')
plt.legend()

print("sum bins", np.sum(binned_delayed))

print("sum bins", np.sum(binned_prompt))

plt.show()
