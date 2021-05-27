import numpy as np
import math
import scipy.linalg as la
from montepython.likelihood_class import Likelihood_prior
from numpy.linalg import multi_dot

class eff_like_from_covmat(Likelihood_prior):
    def __init__(self, path, data, command_line):
        # Call __init__ method of super class:
        super(eff_like_from_covmat, self).__init__(path, data, command_line)
        self.covmat_inverse = la.inv(self.covmat)           
        self.need_cosmo_arguments(data,{'compute damping scale':'yes'})

    # Compute likelihood
    def loglkl(self, cosmo, data):
        covmat = self.covmat
        mu = self.mu
        mean_vec = np.array(mu)
        mean_class = [cosmo.theta_s_100(),cosmo.get_current_derived_parameters(['100*theta_d'])['100*theta_d']]
        mean_vec_class = np.array([eval(s) for s in mean_class])
        #mean_vec_class = np.array([1.04219541,0.32418698]) gives constant likelihood
        dif_vec = mean_vec - mean_vec_class
        dif_vec_T = dif_vec.T
        exponent = dif_vec_T.dot(self.covmat_inverse).dot(dif_vec)
        loglikelihood = -1/2*exponent # exponent in the multivariate Gaussian
        return loglikelihood







