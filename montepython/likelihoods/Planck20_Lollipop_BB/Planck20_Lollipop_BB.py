from montepython.likelihood_class import Likelihood
import numpy as np
import planck_2020_lollipop
import os
import tempfile

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Lollipop_packages"
)

hartlap_factor = False
marginalised_over_covariance = True


class Planck20_Lollipop_BB(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        #Create Cobaya likelihood
        self.lik = planck_2020_lollipop.lowlB({"packages_path": packages_path, "Nsim": self.Nsim, "lmin":self.lmin, "lmax":self.lmax})

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'pCl lCl', 'non_linear': 'halofit', 'l_max_scalars': self.lik.bins.lmax})

        print( "Init Lollipop (lowlB) done !")

    def loglkl(self, cosmo, data):

        print( {par:data.mcmc_parameters[par]['current'] for par in data.get_mcmc_parameters(['cosmo'])})
        cls = self.get_cl(cosmo)
        
        data_params = {par:data.mcmc_parameters[par]['current'] for par in data.get_mcmc_parameters(['nuisance'])}
        print(data_params)
        
        lkl = self.lik.loglike(cls, **data_params)
        print( lkl)
        
        lkl = self.add_nuisance_prior(lkl, data)
        
        return lkl


    def add_nuisance_prior(self, lkl, data):
        # Recover the current value of the nuisance parameter.
        for nuisance in self.use_nuisance:
            nuisance_value = float(
                data.mcmc_parameters[nuisance]['current'] *
                data.mcmc_parameters[nuisance]['scale'])

            # add prior on nuisance parameters
            if hasattr(self, "%s_prior_center" % nuisance) and getattr(self, "%s_prior_std" % nuisance) > 0:
                # convenience variables
                prior_center = getattr(self, "%s_prior_center" % nuisance)
                prior_std = getattr(self, "%s_prior_std" % nuisance)
                lkl += -0.5*((nuisance_value-prior_center)/prior_std)**2
        return lkl


