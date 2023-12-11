from montepython.likelihood_class import Likelihood
import numpy as np
import planck_2020_hillipop
import os
import tempfile

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Hillipop_packages"
)

class Planck20_Hillipop_TTTEEE(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        #Create Cobaya likelihood        
        self.lik = planck_2020_hillipop.TTTEEE({"packages_path": packages_path})

        self.need_cosmo_arguments(
            data, {'lensing': 'yes', 'output': 'tCl pCl lCl', 'non_linear': 'halofit', 'l_max_scalars': self.lik.lmax})

        print( "Init Hillipop TTTEEE done !")

    def loglkl(self, cosmo, data):

        cls = self.get_cl(cosmo)

        fac = cls['ell'] * (cls['ell']+1) / (2*np.pi)
        dl = {mode:np.zeros(self.lik.lmax+1) for mode in ['TT','TE','EE']}
        for mode in ['TT','TE','EE']:
            dl[mode][cls['ell']] = fac*cls[mode.lower()]

        data_params = {par:data.mcmc_parameters[par]['current'] for par in data.get_mcmc_parameters(['nuisance'])}
        for par in ['pe100A','pe100B','pe143A','pe143B']: data_params[par] = 1.0
        data_params['pe217A'] = data_params['pe217B'] = 0.975

        print(data_params)
        
        lkl = self.lik.loglike(dl, **data_params)
        
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
