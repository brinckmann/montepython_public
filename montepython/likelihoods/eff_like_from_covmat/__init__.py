import numpy as np
import math
import scipy.linalg as la
from montepython.likelihood_class import Likelihood_prior
from numpy.linalg import multi_dot


class eff_like_from_covmat(Likelihood_prior):
    def __init__(self, path, data, command_line):
        # Call __init__ method of super class:
        super(eff_like_from_covmat, self).__init__(path, data, command_line)

        self.need_cosmo_arguments(data,{'compute damping scale':'yes'})

    # Compute likelihood
    def loglkl(self, cosmo, data):

        covmat = self.covmat
        mu = self.mu
        mean_vec = np.array(mu)
        covmat_inverse = la.inv(covmat)
        mean_vec_class = [1.042195411207378,0.324186976388346] # 100*theta_s and 100*theta_d from CLASS
        dif_vec = mean_vec - mean_vec_class
        dif_vec_T = dif_vec.T
        #exponent = la.multi_dot([dif_vec.T,covmat_inverse,dif_vec])
        #covmat_inverse = la.inv(self.covmat) # might delete covmat_inverse above
        exponent = dif_vec_T.dot(covmat_inverse).dot(dif_vec)
        loglikelihood = -1/2*exponent # exponent in the multivariate Gaussian
        return loglikelihood







