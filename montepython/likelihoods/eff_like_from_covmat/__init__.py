import numpy as np
import math
import scipy.linalg as la
from montepython.likelihood_class import Likelihood_sn


class eff_like_from_covmat(Likelihood_sn):
    def __init__(self, path, data, command_line):
        # Call __init__ method of super class:
        super(eff_like_from_covmat, self).__init__(path, data, command_line)

        self.need_cosmo_arguments(data,{'compute damping scale':'yes'})
        # Initialise other things:
        covmat_inverse = la.inv(self.covmat)

    # Compute likelihood
    def loglkl(self, cosmo, data):
         
        mean_vec = np.array([self.mu]).T
        mean_vec_class = np.array([eval(s) for s in self.get_var_strings])                                           

        dif_vec = mean_vec - mean_vec_class
        exponent = la.multi_dot([dif_vec.T,covmat_inverse,dif_vec])

        loglikelihood = -1/2*exponent # exponent in the multivariate Gaussian
        return loglikelihood







