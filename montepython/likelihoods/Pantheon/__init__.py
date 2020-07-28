"""
.. module:: Pantheon
    :synopsis: Pantheon likelihood from Jones et al. 2018 and Scolnic et al. 2018

.. moduleauthor:: Rodrigo von Marttens <rodrigovonmarttens@gmail.com> and Antonella Cid <antonellacidm@gmail.com>

Based on the previous JLA likelihood writted by Benjamin Audren

.. code::

    C00 = mag_covmat_file
    
.. note::

    Since there are a lot of file manipulation involved, the "pandas" library
    has to be installed -- it is an 8-fold improvement in speed over numpy, and
    a 2-fold improvement over a fast Python implementation. The "numexpr"
    library is also needed for doing the fast array manipulations, done with
    blas daxpy function in the original c++ code. Both can be installed with
    pip (Python package manager) easily.

"""
import numpy as np
import scipy.linalg as la
import montepython.io_mp as io_mp
try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "(sudo) pip install numexpr --user")
from montepython.likelihood_class import Likelihood_sn


class Pantheon(Likelihood_sn):

    def __init__(self, path, data, command_line):

        # Unusual construction, since the data files are not distributed
        # alongside Pantheon (size problems)
        try:
            Likelihood_sn.__init__(self, path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The Pantheon data files were not found. Please check if "
                "the following files are in the data/Pantheon directory: "
                "\n-> pantheon.dataset"
                "\n-> lcparam_full_long.txt"
                "\n-> sys_full_long.dat")

        # Load matrices from text files, whose names were read in the
        # configuration file
        self.C00 = self.read_matrix(self.mag_covmat_file)
        
        # Reading light-curve parameters from self.data_file (lcparam_full_long.txt)
        self.light_curve_params = self.read_light_curve_parameters()

        # (JR) the following steps can be computed in the initialisation step
        #      as they do not depend on the point in parameter-space
        #   -> likelihood evaluation is 30% faster
        # Compute the covariance matrix
        # The module numexpr is used for doing quickly the long multiplication
        # of arrays (factor of 3 improvements over numpy). It is used as a
        # replacement of blas routines cblas_dcopy and cblas_daxpy
        # For numexpr to work, we need (seems like a bug, but anyway) to create
        # local variables holding the arrays. This cost no time (it is a simple
        # pointer assignment)
        C00 = self.C00
        covm = ne.evaluate("C00")

        sn = self.light_curve_params

        # Update the diagonal terms of the covariance matrix with the
        # statistical error
        covm += np.diag(sn.dmb**2)

        # Whiten the residuals, in two steps
        # 1) Compute the Cholesky decomposition of the covariance matrix, in
        # place. This is a time expensive (0.015 seconds) part -> (JR) do it 
        # in init step, then! needed to be done for each point in JLA likelihood, 
        # but here alpha and beta are no nuisance params anymore and the covmat does
        # not change  
        self.cov = la.cholesky(covm, lower=True, overwrite_a=True)
        # step 2) depends on point in parameter space -> done in loglkl calculation

    def loglkl(self, cosmo, data):
        """
        Compute negative log-likelihood (eq.15 Betoule et al. 2014)

        """
        # Recover the distance moduli from CLASS (a size N vector of double
        # containing the predicted distance modulus for each SN in the JLA
        # sample, given the redshift of the supernova.)
        redshifts = self.light_curve_params.zcmb
        size = redshifts.size

        moduli = np.empty((size, ))
        for index, row in self.light_curve_params.iterrows():
            z_cmb = row['zcmb']
            moduli[index] = cosmo.luminosity_distance(z_cmb)
        moduli = 5 * np.log10(moduli) + 25

        # Convenience variables: store the nuisance parameters in short named
        # variables
        M = (data.mcmc_parameters['M']['current'] *
             data.mcmc_parameters['M']['scale'])

        # Compute the residuals (estimate of distance moduli - exact moduli)
        residuals = np.empty((size,))
        sn = self.light_curve_params
        # This operation loops over all supernovae!
        # Compute the approximate moduli
        residuals = sn.mb - M
        # Remove from the approximate moduli the one computed from CLASS
        residuals -= moduli

        # 2) Solve the triangular system, also time expensive (0.02 seconds)
        residuals = la.solve_triangular(self.cov, residuals, lower=True, check_finite=False)

        # Finally, compute the chi2 as the sum of the squared residuals
        chi2 = (residuals**2).sum()

        return -0.5 * chi2
