from unittest import TestCase, main



from pyCEvNS.flux import NeutrinoFluxFactory
# neutrino flux factory
class NeutrinoTestCase(TestCase):
    def test_flux_factory(self):
        # check they all load properly
        nuff = NeutrinoFluxFactory()
        for name in nuff.flux_list:
            nuff.get(name, zenith=0.025)
    
    def test_neutrino_flux(self):
        coh = NeutrinoFluxFactory().get("coherent")
        assert(coh.integrate(0, 200, "mu") > 0)



# dm flux factory



# axion flux factory
from pyCEvNS.axion import PrimakoffAxionFromBeam

class AxionTestCase(TestCase):
    def test_null_photon_flux(self):
        # Check that a [[]] flux passes in okay
        alp_from_beam = PrimakoffAxionFromBeam(photon_rates=[])
        alp_from_beam.simulate()
    def test_kinematically_blocked(self):
        # Check that if E_gamma < m_a, we generate no events.
        alp_from_beam = PrimakoffAxionFromBeam(photon_rates=[[1,1,0]], axion_mass=5)
        alp_from_beam.simulate()
        assert(len(alp_from_beam.axion_energy)==0)
    def test_good_scatter(self):
        alp_from_beam = PrimakoffAxionFromBeam(photon_rates=[[1,1,0]], axion_mass=0.001, axion_coupling=1e-3)
        alp_from_beam.simulate()
        assert(alp_from_beam.scatter_events(detector_number=1e10, detector_z=20, detection_time=10, threshold=0) > 0.)








if __name__ == "__main__":
    main()