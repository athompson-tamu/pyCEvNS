from pyCEvNS.flux import *

class DMFluxFromPhoton(FluxBaseContinuous):
    def __init__(self, photon_distribution, dark_photon_mass, coupling, dark_matter_mass,
                 detector_distance=19.3, detector_direction=0, detector_width=0.1, pot_rate=5e20, pot_sample=100000,
                 pot_mu=0.7, pot_sigma=0.15, photon_rate=1/100000, exp=4466, sampling_size=100, nbins=50, verbose=False):
        self.dp_m = dark_photon_mass
        self.epsilon = coupling
        self.dm_m = dark_matter_mass
        self.det_dist = detector_distance
        self.det_direc = detector_direction
        self.det_width = detector_width
        self.pot_rate = pot_rate  # the number of POT/day in the experiment
        self.pot_mu = pot_mu * 1e-6
        self.pot_sigma = pot_sigma * 1e-6
        self.pot_sample = pot_sample  # the number of POT corresponding to the supplied photon_distribution sample
        self.photon_rate = photon_rate
        self.exp = exp
        self.time = []
        self.energy = []
        self.weight = []
        self.dm_mass = dark_matter_mass
        self.sampling_size = sampling_size
        self.verbose = verbose
        for photon_events in photon_distribution:
            if self.verbose:
                print("getting photons from E =", photon_events[0], "Size =", photon_events[1])
            self._simulate_dm_events(photon_events, self.sampling_size)
        self.timing = np.array(self.time)*1e6
        hist, bin_edges = np.histogram(self.energy, bins=nbins, density=True)
        super().__init__((bin_edges[:-1]+bin_edges[1:])/2, hist,
                         norm=self.epsilon**2*pot_rate*photon_rate*len(self.time)/len(photon_distribution)
                              /(4 * np.pi * (self.det_dist ** 2) * 24 * 3600)*meter_by_mev**2/self.sampling_size)


    def events_yield(self):
        normalization = self.exp * self.epsilon**2 * \
                        (self.pot_rate / self.pot_sample) / (4 * np.pi * (self.det_dist ** 2))
        sum_of_weights = np.sum(self.weight) * normalization
        return  sum_of_weights


    def _simulate_dm_events(self, photon_events, nsamples):
        # Initiate photon position, energy and momentum.
        pos = np.zeros(3)
        dp_e = photon_events[0]
        dp_p = np.sqrt(photon_events[0]**2-self.dp_m**2)
        dp_momentum = np.array([dp_e, 0, 0, dp_p])

        # dark photon to dark matter
        dm_m = self.dm_m
        dm_e = self.dp_m / 2
        dm_p = np.sqrt(dm_e**2 - dm_m**2)

        # Directional sampling.
        for i in range(0, nsamples):
            dp_wgt = photon_events[1] / nsamples
            csd = np.random.uniform(-1, 1)
            phid = np.random.uniform(0, 2*np.pi)
            dm_momentum = np.array([dm_e, dm_p*np.sqrt(1-csd**2)*np.cos(phid),
                                    dm_p*np.sqrt(1-csd**2)*np.sin(phid), dm_p*csd])
            dm_momentum = lorentz_boost(dm_momentum,
                                        np.array([-dp_momentum[1]/dp_momentum[0],
                                                  -dp_momentum[2]/dp_momentum[0],
                                                  -dp_momentum[3]/dp_momentum[0]]))

            # dark matter arrives at detector, assuming azimuthal symmetric
            # append the time and energy spectrum of the DM.
            v = dm_momentum[1:]/dm_momentum[0]*c_light
            a = np.sum(v**2)
            b = 2*np.sum(v)
            c = np.sum(pos**2) - self.det_dist**2
            if b**2 - 4*a*c >= 0:
                t_dm = (-b+np.sqrt(b**2-4*a*c))/(2*a)
                if t_dm >= 0 and self.det_direc-self.det_width/2 <= \
                    (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t_dm)
                    self.energy.append(dm_momentum[0])
                    self.weight.append(dp_wgt)
                t_dm = (-b-np.sqrt(b**2-4*a*c))/(2*a)
                if t_dm >= 0 and self.det_direc-self.det_width/2 <= \
                    (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t_dm)
                    self.energy.append(dm_momentum[0])
                    self.weight.append(dp_wgt)
            v = (dp_momentum-dm_momentum)[1:]/(dp_momentum-dm_momentum)[0]*c_light
            a = np.sum(v**2)
            b = 2*np.sum(v)
            c = np.sum(pos**2) - self.det_dist**2
            if b**2 - 4*a*c >= 0:
                t_dm = (-b+np.sqrt(b**2-4*a*c))/(2*a)
                if t_dm >= 0 and self.det_direc-self.det_width/2 <= \
                    (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t_dm)
                    self.energy.append((dp_momentum-dm_momentum)[0])
                    self.weight.append(dp_wgt)
                t_dm = (-b-np.sqrt(b**2-4*a*c))/(2*a)
                if t_dm >= 0 and self.det_direc-self.det_width/2 <= \
                    (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t_dm)
                    self.energy.append((dp_momentum-dm_momentum)[0])
                    self.weight.append(dp_wgt)