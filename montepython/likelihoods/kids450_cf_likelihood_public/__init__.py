##############################################################
# likelihood for the KiDS-450 correlation functions          #
##############################################################
#
# Originally set up by Antonio J. Cuesta and J. Lesgourgues
# for CFHTLenS data, by adapting Benjamin Audren's Monte Python
# likelihood euclid_lensing and Adam J Moss's CosmoMC likelihood
# for weak lensing (adapted itself from JL's CosmoMC likelihood
# for the COSMOS).
#
# Adjusted for KiDS-450 correlation function data from Hildebrandt
# et al. 2017 (MNRAS, 465, 1454) by Fabian Koehlinger.
#
# Data available from:
#
# http://kids.strw.leidenuniv.nl/sciencedata.php
#
# ATTENTION:
# 1) This is NOT the likelihood used in their original analysis
# and the major difference being the usage of HMcode (in the
# original analysis) vs. a fitting formula for baryon feedback (as
# used in this likelihood). However, the difference was found to be
# negligible in the analysis of Koehlinger et al. 2019 (MNRAS, 484,
# 3126).
# 2) This likelihood only produces valid results for \Omega_k = 0,
# i.e. flat cosmologies!
##############################################################
from __future__ import print_function

from montepython.likelihood_class import Likelihood
import io_mp
import parser_mp

#import scipy.integrate
from scipy import interpolate as itp
from scipy import special
from scipy.linalg import cholesky, solve_triangular
import os
import numpy as np
import math

try:
    xrange
except NameError:
    xrange = range

class kids450_cf_likelihood_public(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Check if the data can be found, although we don't actually use that
        # particular file but take it as a placeholder for the folder
        try:
            fname = os.path.join(self.data_directory, 'DATA_VECTOR/KiDS-450_xi_pm_tomographic_data_vector.dat')
            parser_mp.existing_file(fname)
        except:
            raise io_mp.ConfigurationError('KiDS-450 CF data not found. Download the data at '
                                           'http://kids.strw.leidenuniv.nl/sciencedata.php '
                                           'and specify path to data through the variable '
                                           'kids450_cf_likelihood_public.data_directory in '
                                           'the .data file. See README in likelihood folder '
                                           'for further instructions.')

        # for loading of Nz-files:
        self.z_bins_min = [0.1, 0.3, 0.5, 0.7]
        self.z_bins_max = [0.3, 0.5, 0.7, 0.9]

        # number of angular bins in which xipm is measured
        # we always load the full data vector with 9 data points for xi_p and
        # xi_m each; they are cut to the fiducial scales (or any arbitrarily
        # defined scales with the 'cut_values.dat' files!
        self.ntheta = 9

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z) and for k up to k_max
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': self.k_max_h_by_Mpc})

        # Compute non-linear power spectrum if requested:
        if self.method_non_linear_Pk in ['halofit', 'HALOFIT', 'Halofit', 'hmcode', 'Hmcode', 'HMcode', 'HMCODE']:
            self.need_cosmo_arguments(data, {'non linear': self.method_non_linear_Pk})
            print('Using {:} to obtain the non-linear P(k, z)!'.format(self.method_non_linear_Pk))
        else:
            print('Only using the linear P(k, z) for ALL calculations \n (check keywords for "method_non_linear_Pk").')

        # Define array of l values, and initialize them
        # It is a logspace
        # find nlmax in order to reach lmax with logarithmic steps dlnl
        self.nlmax = np.int(np.log(self.lmax) / self.dlnl) + 1
        # redefine slightly dlnl so that the last point is always exactly lmax
        self.dlnl = np.log(self.lmax) / (self.nlmax - 1)
        self.l = np.exp(self.dlnl * np.arange(self.nlmax))

        self.nzbins = len(self.z_bins_min)
        self.nzcorrs = self.nzbins * (self.nzbins + 1) // 2

        # Create labels for loading of dn/dz-files:
        self.zbin_labels = []
        for i in xrange(self.nzbins):
            self.zbin_labels += ['{:.1f}t{:.1f}'.format(self.z_bins_min[i], self.z_bins_max[i])]

        # read in public data vector:
        temp = self.__load_public_data_vector()
        self.theta_bins = temp[:, 0]
        if (np.sum(
                (self.theta_bins[:self.ntheta] -
                    self.theta_bins[self.ntheta:])**2) > 1e-6):
                raise io_mp.LikelihoodError(
                    'The angular values at which xi+ and xi- '
                    'are observed do not match')

        # create the data-vector in the following format (due to covariance structure):
        # xi_obs = {xi1(theta1, z_11)...xi1(theta_k, z_11), xi2(theta_1, z_11)...
        #           xi2(theta_k, z_11);...; xi1(theta1, z_nn)...xi1(theta_k, z_nn),
        #           xi2(theta_1, z_nn)... xi2(theta_k, z_nn)}
        self.xi_obs = self.__get_xi_obs(temp[:, 1:])

        # now load the full covariance matrix:
        covmat = self.__load_public_cov_mat()

        # Read angular cut values (OPTIONAL)
        if self.use_cut_theta:
            cut_values1 = np.zeros((self.nzbins, 2))
            cut_values2 = np.zeros((self.nzbins, 2))

            cutvalues_file_path1 = os.path.join(self.data_directory, 'CUT_VALUES/' + self.cutvalues_file1)
            if os.path.exists(cutvalues_file_path1):
                cut_values1 = np.loadtxt(cutvalues_file_path1)
            else:
                raise io_mp.LikelihoodError('File not found:\n {:} \n Check that requested file was copied to:\n {:}'.format(cutvalues_file_path1, self.data_directory + 'CUT_VALUES/'))

            if self.subtract_mask2_from_mask1:
                cutvalues_file_path2 = os.path.join(self.data_directory, 'CUT_VALUES/' + self.cutvalues_file2)
                if os.path.exists(cutvalues_file_path2):
                    cut_values2 = np.loadtxt(cutvalues_file_path2)
                else:
                    raise io_mp.LikelihoodError('File not found:\n {:} \n Check that requested file was copied to:\n {:}'.format(cutvalues_file_path2, self.data_directory + 'CUT_VALUES/'))

        # Compute theta mask
        if self.use_cut_theta:
            mask1 = self.__get_mask(cut_values1)
            if self.subtract_mask2_from_mask1:
                mask2 = self.__get_mask(cut_values2)
                mask = mask1 - mask2
            else:
                mask = mask1
        else:
            mask = np.ones(2 * self.nzcorrs * self.ntheta)

        self.mask_indices = np.where(mask == 1)[0]
        fname = os.path.join(self.data_directory, 'kids450_xipm_4bin_cut.dat')
        np.savetxt(fname, self.xi_obs[self.mask_indices])

        # propagate uncertainty of m-correction following equation (12) in
        # Hildebrandt et al. 2017 (arXiv:1606.05338) with \sigma_m = 0.01
        # NOTE: following Troxel et al. 2018 (arXiv:1804.10663) it is NOT
        # correct to use the noisy data vector for this; instead one should use
        # a theory vector (e.g. derived for the same cosmology for which the
        # analytical covariance was calculated).
        fname = os.path.join(self.data_directory, 'cov_matrix_ana_cut.dat')
        if self.marginalize_over_multiplicative_bias_uncertainty:
            cov_m_corr = np.matrix(self.xi_obs[self.mask_indices]).T * np.matrix(self.xi_obs[self.mask_indices]) * 4. * self.err_multiplicative_bias**2
            covmat = covmat[self.mask_indices][:, self.mask_indices] + np.asarray(cov_m_corr)
            #covmat = covmat[np.ix_(self.mask_indices, self.mask_indices)]
            np.savetxt(fname, covmat)
            #covmat = covmat + np.asarray(cov_m_corr)
        else:
            #covmat = covmat[self.mask_indices][:, self.mask_indices]
            covmat = covmat[np.ix_(self.mask_indices, self.mask_indices)]
            np.savetxt(fname, covmat)

        # precompute Cholesky transform for chi^2 calculation:
        self.cholesky_transform = cholesky(covmat, lower=True)

        # Read fiducial dn_dz from window files:
        #self.z_p = np.zeros(self.nzmax)
        # TODO: the hardcoded z_min and z_max correspond to the lower and upper
        # endpoints of the shifted left-border histogram!
        self.z_p = np.linspace(0.025, 3.475, self.nzmax)
        self.pz = np.zeros((self.nzmax, self.nzbins))
        self.pz_norm = np.zeros(self.nzbins, 'float64')
        for zbin in xrange(self.nzbins):
            window_file_path = os.path.join(
                self.data_directory, 'Nz_{0:}/Nz_{0:}_Mean/Nz_{0:}_z{1:}.asc'.format(self.nz_method, self.zbin_labels[zbin]))

            zptemp, hist_pz = np.loadtxt(window_file_path, usecols=[0, 1], unpack=True)
            if zbin > 0:
                zpcheck = zptemp
                if np.sum((zptemp - zpcheck)**2) > 1e-6:
                    raise io_mp.LikelihoodError('The redshift values for the window files at different bins do not match.')
            print('Loaded n(zbin{:}) from: \n'.format(zbin + 1), window_file_path)
            # we assume that the histograms loaded are given as left-border histograms
            # and that the z-spacing is the same for each histogram
            shift_to_midpoint = np.diff(zptemp)[0] / 2.
            spline_pz = itp.splrep(zptemp + shift_to_midpoint, hist_pz)
            z_mod = self.z_p #+ shift_by_dz[zbin]
            mask_min = z_mod >= zptemp.min()
            mask_max = z_mod <= zptemp.max()
            mask = mask_min & mask_max
            # points outside the z-range of the histograms are set to 0!
            self.pz[mask, zbin] = itp.splev(z_mod[mask], spline_pz)
            # Normalize selection functions
            dz = self.z_p[1:] - self.z_p[:-1]
            self.pz_norm[zbin] = np.sum(0.5 * (self.pz[1:, zbin] + self.pz[:-1, zbin]) * dz)

        self.zmax = self.z_p.max()
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})

        ################################################
        # discrete theta values (to convert C_l to xi's)
        ################################################

        thetamin = np.min(self.theta_bins) * 0.8
        thetamax = np.max(self.theta_bins) * 1.2

        self.nthetatot = np.ceil(math.log(thetamax / thetamin) / self.dlntheta) + 1
        self.nthetatot = np.int32(self.nthetatot)
        self.theta = np.zeros(self.nthetatot, 'float64')
        self.a2r = math.pi / (180. * 60.)

        # define an array of theta's
        for it in xrange(self.nthetatot):
            self.theta[it] = thetamin * math.exp(self.dlntheta * it)

        ################################################################
        # discrete l values used in the integral to convert C_l to xi's)
        ################################################################

        # l = x / theta / self.a2r
        # x = l * theta * self.a2r

        # We start by considering the largest theta, theta[-1], and for that value we infer
        # a list of l's from the requirement that corresponding x values are spaced linearly with a given stepsize, until xmax.
        # Then we loop over smaller theta values, in decreasing order, and for each of them we complete the previous list of l's,
        # always requiuring the same dx stepsize (so that dl does vary) up to xmax.
        #
        # We first apply this to a running value ll, in order to count the total numbner of ll's, called nl.
        # Then we create the array lll[nl] and we fill it with the same values.
        #
        # we also compute on the fly the critical index il_max[it] such that ll[il_max[it]]*self.theta[it]*self.a2r
        # is the first value of x above xmax

        ll=1.
        il=0
        while (ll*self.theta[-1]*self.a2r < self.dx_threshold):
            ll += self.dx_below_threshold/self.theta[-1]/self.a2r
            il += 1
        for it  in xrange(self.nthetatot):
            while (ll*self.theta[self.nthetatot-1-it]*self.a2r < self.xmax) and (ll+self.dx_above_threshold/self.theta[self.nthetatot-1-it]/self.a2r < self.lmax):
                ll += self.dx_above_threshold/self.theta[self.nthetatot-1-it]/self.a2r
                il += 1
        self.nl = il+1

        self.lll = np.zeros(self.nl, 'float64')
        self.il_max = np.zeros(self.nthetatot, 'int')
        il=0
        self.lll[il]=1.
        while (self.lll[il]*self.theta[-1]*self.a2r < self.dx_threshold):
            il += 1
            self.lll[il] = self.lll[il-1] + self.dx_below_threshold/self.theta[-1]/self.a2r
        for it  in xrange(self.nthetatot):
            while (self.lll[il]*self.theta[self.nthetatot-1-it]*self.a2r < self.xmax) and (self.lll[il] + self.dx_above_threshold/self.theta[self.nthetatot-1-it]/self.a2r < self.lmax):
                il += 1
                self.lll[il] = self.lll[il-1] + self.dx_above_threshold/self.theta[self.nthetatot-1-it]/self.a2r
            self.il_max[self.nthetatot-1-it] = il

        # finally we compute the array l*dl that will be used in the trapezoidal integration
        # (l is a factor in the integrand [l * C_l * Bessel], and dl is like a weight)
        self.ldl = np.zeros(self.nl, 'float64')
        self.ldl[0]=self.lll[0]*0.5*(self.lll[1]-self.lll[0])
        for il in xrange(1,self.nl-1):
            self.ldl[il]=self.lll[il]*0.5*(self.lll[il+1]-self.lll[il-1])
        self.ldl[-1]=self.lll[-1]*0.5*(self.lll[-1]-self.lll[-2])

        #####################################################################
        # Allocation of various arrays filled and used in the function loglkl
        #####################################################################

        self.r = np.zeros(self.nzmax, 'float64')
        self.dzdr = np.zeros(self.nzmax, 'float64')
        self.g = np.zeros((self.nzmax, self.nzbins), 'float64')
        self.pk = np.zeros((self.nlmax, self.nzmax), 'float64')
        self.k_sigma = np.zeros(self.nzmax, 'float64')
        self.alpha = np.zeros((self.nlmax, self.nzmax), 'float64')
        if 'epsilon' in self.use_nuisance:
            self.E_th_nu = np.zeros((self.nlmax, self.nzmax), 'float64')
        self.Cl_integrand = np.zeros((self.nzmax, self.nzcorrs), 'float64')
        self.Cl = np.zeros((self.nlmax, self.nzcorrs), 'float64')
        '''
        if self.theoretical_error != 0:
            self.El_integrand = np.zeros((self.nzmax, self.nzcorrs),'float64')
            self.El = np.zeros((self.nlmax, self.nzcorrs), 'float64')
        '''
        self.spline_Cl = np.empty(self.nzcorrs, dtype=(list, 3))
        self.xi1 = np.zeros((self.nthetatot, self.nzcorrs), 'float64')
        self.xi2 = np.zeros((self.nthetatot, self.nzcorrs), 'float64')
        self.Cll = np.zeros((self.nzcorrs, self.nl), 'float64')
        self.BBessel0 = np.zeros(self.nl, 'float64')
        self.BBessel4 = np.zeros(self.nl, 'float64')
        self.xi1_theta = np.empty(self.nzcorrs, dtype=(list, 3))
        self.xi2_theta = np.empty(self.nzcorrs, dtype=(list, 3))
        self.xi = np.zeros(np.size(self.xi_obs), 'float64')

        return


    def __load_public_data_vector(self):
        """
        Helper function to read in the full data vector from public KiDS-450
        release and to bring it into the input format used in the original
        CFHTLenS likelihood.
        """
        # plus one for theta-column
        data_xip = np.zeros((self.ntheta, self.nzcorrs + 1))
        data_xim = np.zeros((self.ntheta, self.nzcorrs + 1))
        idx_corr = 0
        for zbin1 in xrange(self.nzbins):
            for zbin2 in xrange(zbin1, self.nzbins):

                fname = os.path.join(self.data_directory, 'DATA_VECTOR/KiDS-450_xi_pm_files/KiDS-450_xi_pm_tomo_{:}_{:}_logbin_mcor.dat'.format(zbin1 + 1, zbin2 + 1))
                theta, xip, xim = np.loadtxt(fname, unpack=True)

                # this assumes theta is the same for every tomographic bin and
                # for both xi_p and xi_m!
                if idx_corr == 0:
                    data_xip[:, 0] = theta
                    data_xim[:, 0] = theta

                data_xip[:, idx_corr + 1] = xip
                data_xim[:, idx_corr + 1] = xim

                idx_corr += 1

        data = np.concatenate((data_xip, data_xim))

        return data


    def __load_public_cov_mat(self):
        """
        Helper function to read in the full covariance matrix from the public
        KiDS-450 release and to bring it into format of self.xi_obs.
        """

        try:
            fname = os.path.join(self.data_directory, 'COV_MAT/Cov_mat_all_scales_use_with_kids450_cf_likelihood_public.dat')
            matrix = np.loadtxt(fname)

        except:
            print('\n Creating covariance matrix for all scales for first time. \n This might take some minutes, but only once...')
            fname = os.path.join(self.data_directory, 'COV_MAT/Cov_mat_all_scales.txt')
            tmp_raw = np.loadtxt(fname)

            thetas_plus = self.theta_bins[:self.ntheta]
            thetas_minus = self.theta_bins[self.ntheta:]

            indices = np.column_stack((tmp_raw[:, :3], tmp_raw[:, 4:7]))

            # we need to add all three components for full covariance
            values = tmp_raw[:, 8] + tmp_raw[:, 9] + tmp_raw[:, 10]

            for i in xrange(len(tmp_raw)):
                for j in xrange(self.ntheta):
                    if np.abs(tmp_raw[i, 3] - thetas_plus[j]) <= 0.05:
                        tmp_raw[i, 3] = j
                    if np.abs(tmp_raw[i, 7] - thetas_plus[j]) <= 0.05:
                        tmp_raw[i, 7] = j

            thetas_raw_plus = tmp_raw[:, 3]
            thetas_raw_minus = tmp_raw[:, 7]

            dim = 2 * self.ntheta * self.nzcorrs
            matrix = np.zeros((dim, dim))

            # ugly brute-force...
            index1 = 0
            # this creates the correctly ordered (i.e. like self.xi_obs) full
            # 180 x 180 covariance matrix:
            for iz1 in xrange(self.nzbins):
                for iz2 in xrange(iz1, self.nzbins):
                    for ipm in xrange(2):
                        for ith in xrange(self.ntheta):

                            index2 = 0
                            for iz3 in xrange(self.nzbins):
                                for iz4 in xrange(iz3, self.nzbins):
                                    for ipm2 in xrange(2):
                                        for ith2 in xrange(self.ntheta):
                                            for index_lin in xrange(len(tmp_raw)):
                                                #print(index1, index2)
                                                #print(iz1, iz2, ipm, ith, iz3, iz4, ipm2)
                                                if iz1 + 1 == indices[index_lin, 0] and iz2 + 1 == indices[index_lin, 1] and ipm == indices[index_lin, 2] and iz3 + 1 == indices[index_lin, 3]  and iz4 + 1 == indices[index_lin, 4] and ipm2 == indices[index_lin, 5] and ith == thetas_raw_plus[index_lin] and ith2 == thetas_raw_minus[index_lin]:
                                                    #print('hit')
                                                    matrix[index1, index2] = values[index_lin]
                                                    matrix[index2, index1] = matrix[index1, index2]
                                            index2 += 1
                            index1 += 1

            fname = fname = os.path.join(self.data_directory, 'COV_MAT/Cov_mat_all_scales_use_with_kids450_cf_likelihood_public.dat')
            if not os.path.isfile(fname):
                np.savetxt(fname, matrix)
                print('Saved covariance matrix structured as needed for this likelihood to: \n', fname)

        return matrix


    def __get_mask(self, cut_values):

        mask = np.zeros(2 * self.nzcorrs * self.ntheta)
        iz = 0
        for izl in xrange(self.nzbins):
            for izh in xrange(izl, self.nzbins):
                # this counts the bin combinations
                # iz=1 =>(1,1), iz=2 =>(1,2) etc
                iz = iz + 1
                for i in xrange(self.ntheta):
                    j = (iz-1)*2*self.ntheta
                    #xi_plus_cut = max(cut_values[izl, 0], cut_values[izh, 0])
                    xi_plus_cut_low = max(cut_values[izl, 0], cut_values[izh, 0])
                    xi_plus_cut_high = max(cut_values[izl, 1], cut_values[izh, 1])
                    #xi_minus_cut = max(cut_values[izl, 1], cut_values[izh, 1])
                    xi_minus_cut_low = max(cut_values[izl, 2], cut_values[izh, 2])
                    xi_minus_cut_high = max(cut_values[izl, 3], cut_values[izh, 3])
                    if ((self.theta_bins[i] < xi_plus_cut_high) and (self.theta_bins[i]>xi_plus_cut_low)):
                        mask[j+i] = 1
                    if ((self.theta_bins[i] < xi_minus_cut_high) and (self.theta_bins[i]>xi_minus_cut_low)):
                        mask[self.ntheta + j+i] = 1

        return mask

    def __get_xi_obs(self, temp):

        xi_obs = np.zeros(self.ntheta * self.nzcorrs * 2)

        # create the data-vector:
        # xi_obs = {xi1(theta1, z_11)...xi1(theta_k, z_11), xi2(theta_1, z_11)...
        #           xi2(theta_k, z_11);...; xi1(theta1, z_nn)...xi1(theta_k, z_nn),
        #           xi2(theta_1, z_nn)... xi2(theta_k, z_nn)}
        k = 0
        for j in xrange(self.nzcorrs):
            for i in xrange(2 * self.ntheta):
                xi_obs[k] = temp[i, j]
                k += 1

        return xi_obs

    def baryon_feedback_bias_sqr(self, k, z, A_bary=1.):
        """

        Fitting formula for baryon feedback after equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

        """

        # k is expected in h/Mpc and is divided in log by this unit...
        x = np.log10(k)

        a = 1. / (1. + z)
        a_sqr = a * a

        constant = {'AGN':   {'A2': -0.11900, 'B2':  0.1300, 'C2':  0.6000, 'D2':  0.002110, 'E2': -2.0600,
                              'A1':  0.30800, 'B1': -0.6600, 'C1': -0.7600, 'D1': -0.002950, 'E1':  1.8400,
                              'A0':  0.15000, 'B0':  1.2200, 'C0':  1.3800, 'D0':  0.001300, 'E0':  3.5700},
                    'REF':   {'A2': -0.05880, 'B2': -0.2510, 'C2': -0.9340, 'D2': -0.004540, 'E2':  0.8580,
                              'A1':  0.07280, 'B1':  0.0381, 'C1':  1.0600, 'D1':  0.006520, 'E1': -1.7900,
                              'A0':  0.00972, 'B0':  1.1200, 'C0':  0.7500, 'D0': -0.000196, 'E0':  4.5400},
                    'DBLIM': {'A2': -0.29500, 'B2': -0.9890, 'C2': -0.0143, 'D2':  0.001990, 'E2': -0.8250,
                              'A1':  0.49000, 'B1':  0.6420, 'C1': -0.0594, 'D1': -0.002350, 'E1': -0.0611,
                              'A0': -0.01660, 'B0':  1.0500, 'C0':  1.3000, 'D0':  0.001200, 'E0':  4.4800}}

        A_z = constant[self.baryon_model]['A2']*a_sqr+constant[self.baryon_model]['A1']*a+constant[self.baryon_model]['A0']
        B_z = constant[self.baryon_model]['B2']*a_sqr+constant[self.baryon_model]['B1']*a+constant[self.baryon_model]['B0']
        C_z = constant[self.baryon_model]['C2']*a_sqr+constant[self.baryon_model]['C1']*a+constant[self.baryon_model]['C0']
        D_z = constant[self.baryon_model]['D2']*a_sqr+constant[self.baryon_model]['D1']*a+constant[self.baryon_model]['D0']
        E_z = constant[self.baryon_model]['E2']*a_sqr+constant[self.baryon_model]['E1']*a+constant[self.baryon_model]['E0']

        # only for debugging; tested and works!
        #print('AGN: A2=-0.11900, B2= 0.1300, C2= 0.6000, D2= 0.002110, E2=-2.0600')
        #print(self.baryon_model+': A2={:.5f}, B2={:.5f}, C2={:.5f}, D2={:.5f}, E2={:.5f}'.format(constant[self.baryon_model]['A2'], constant[self.baryon_model]['B2'], constant[self.baryon_model]['C2'],constant[self.baryon_model]['D2'], constant[self.baryon_model]['E2']))

        # original formula:
        #bias_sqr = 1.-A_z*np.exp((B_z-C_z)**3)+D_z*x*np.exp(E_z*x)
        # original formula with a free amplitude A_bary:
        bias_sqr = 1. - A_bary * (A_z * np.exp((B_z * x - C_z)**3) - D_z * x * np.exp(E_z * x))

        return bias_sqr

    def get_IA_factor(self, z, linear_growth_rate, amplitude, exponent):

        const = 5e-14 / self.small_h**2 # Mpc^3 / M_sol

        # arbitrary convention
        z0 = 0.3
        #print(utils.growth_factor(z, self.Omega_m))
        #print(self.rho_crit)
        factor = -1. * amplitude * const * self.rho_crit * self.Omega_m / linear_growth_rate * ((1. + z) / (1. + z0))**exponent

        return factor

    def get_critical_density(self):
        """
        The critical density of the Universe at redshift 0.

        Returns
        -------
        rho_crit in solar masses per cubic Megaparsec.

        """

        # yay, constants...
        Mpc_cm = 3.08568025e24 # cm
        M_sun_g = 1.98892e33 # g
        G_const_Mpc_Msun_s = M_sun_g * (6.673e-8) / Mpc_cm**3.
        H100_s = 100. / (Mpc_cm * 1.0e-5) # s^-1

        rho_crit_0 = 3. * (self.small_h * H100_s)**2. / (8. * np.pi * G_const_Mpc_Msun_s)

        return rho_crit_0


    def loglkl(self, cosmo, data):

        # Omega_m contains all species!
        self.Omega_m = cosmo.Omega_m()
        self.small_h = cosmo.h()

        # needed for IA modelling:
        if ('A_IA' in data.mcmc_parameters) and ('exp_IA' in data.mcmc_parameters):
            amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            exp_IA = data.mcmc_parameters['exp_IA']['current'] * data.mcmc_parameters['exp_IA']['scale']
            intrinsic_alignment = True
        elif ('A_IA' in data.mcmc_parameters) and ('exp_IA' not in data.mcmc_parameters):
            amp_IA = data.mcmc_parameters['A_IA']['current'] * data.mcmc_parameters['A_IA']['scale']
            # redshift-scaling is turned off:
            exp_IA = 0.

            intrinsic_alignment = True
        else:
            intrinsic_alignment = False

        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        self.r, self.dzdr = cosmo.z_of_r(self.z_p)

        # Compute now the selection function p(r) = p(z) dz/dr normalized
        # to one. The np.newaxis helps to broadcast the one-dimensional array
        # dzdr to the proper shape. Note that p_norm is also broadcasted as
        # an array of the same shape as p_z
        if self.bootstrap_photoz_errors:
            # draw a random bootstrap n(z); borders are inclusive!
            random_index_bootstrap = np.random.randint(int(self.index_bootstrap_low), int(self.index_bootstrap_high) + 1)
            #print('Bootstrap index:', random_index_bootstrap)
            pz = np.zeros((self.nzmax, self.nzbins), 'float64')
            pz_norm = np.zeros(self.nzbins, 'float64')

            for zbin in xrange(self.nzbins):
                #ATTENTION: hard-coded subfolder!
                #index can be recycled since bootstraps for tomographic bins are independent!
                fname = os.path.join(self.data_directory, 'Nz_{0:}/Nz_{0:}_Bootstrap/Nz_z{1:}_boot{2:}_{0:}.asc'.format(self.nz_method, self.zbin_labels[zbin], random_index_bootstrap))
                ztemp, hist_pz = np.loadtxt(fname, usecols=(0, 1), unpack=True)
                shift_to_midpoint = np.diff(ztemp)[0] / 2.
                spline_pz = itp.splrep(ztemp + shift_to_midpoint, hist_pz)
                z_mod = self.z_p #+ self.shift_by_dz[zbin]
                mask_min = z_mod >= ztemp.min()
                mask_max = z_mod <= ztemp.max()
                mask = mask_min & mask_max
                pz[mask, zbin] = itp.splev(z_mod[mask], spline_pz)
                pz_norm[zbin] = np.sum(0.5 * (pz[1:, zbin] + pz[:-1, zbin]) * (self.z_p[1:] - self.z_p[:-1]))

            self.pr = pz * (self.dzdr[:, np.newaxis] / pz_norm)
        else:
            # use fiducial dn/dz loaded in the __init__:
            self.pr = self.pz * (self.dzdr[:, np.newaxis] / self.pz_norm)

        # get linear growth rate if IA are modelled:
        if intrinsic_alignment:
            self.rho_crit = self.get_critical_density()
            # derive the linear growth factor D(z)
            linear_growth_rate = np.zeros_like(self.z_p)
            #print(self.redshifts)
            for index_z, z in enumerate(self.z_p):
                try:
                    # for CLASS ver >= 2.6:
                    linear_growth_rate[index_z] = cosmo.scale_independent_growth_factor(z)
                except:
                    # my own function from private CLASS modification:
                    linear_growth_rate[index_z] = cosmo.growth_factor_at_z(z)
            # normalize to unity at z=0:
            try:
                # for CLASS ver >= 2.6:
                linear_growth_rate /= cosmo.scale_independent_growth_factor(0.)
            except:
                # my own function from private CLASS modification:
                linear_growth_rate /= cosmo.growth_factor_at_z(0.)

        # Compute function g_i(r), that depends on r and the bin
        # g_i(r) = 2r(1+z(r)) int_r^+\infty drs p_r(rs) (rs-r)/rs
        for Bin in xrange(self.nzbins):
            # shift from first entry only useful if first enrty is 0!
            #for nr in xrange(1, self.nzmax-1):
            for nr in xrange(self.nzmax - 1):
                fun = self.pr[nr:, Bin] * (self.r[nr:] - self.r[nr]) / self.r[nr:]
                self.g[nr, Bin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (self.r[nr + 1:] - self.r[nr:-1]))
                self.g[nr, Bin] *= 2. * self.r[nr] * (1. + self.z_p[nr])
        #print('g(r) \n', self.g)

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        #self.pk_dm = np.zeros_like(self.pk)
        kmax_in_inv_Mpc = self.k_max_h_by_Mpc * self.small_h
        for index_l in xrange(self.nlmax):
            for index_z in xrange(self.nzmax):

                k_in_inv_Mpc =  (self.l[index_l] + 0.5) / self.r[index_z]
                if (k_in_inv_Mpc > kmax_in_inv_Mpc):
                    pk_dm = 0.
                else:
                    pk_dm = cosmo.pk(k_in_inv_Mpc, self.z_p[index_z])

                if 'A_bary' in data.mcmc_parameters:
                    A_bary = data.mcmc_parameters['A_bary']['current'] * data.mcmc_parameters['A_bary']['scale']
                    self.pk[index_l, index_z] = pk_dm * self.baryon_feedback_bias_sqr(k_in_inv_Mpc / self.small_h, self.z_p[index_z], A_bary=A_bary)
                else:
                    self.pk[index_l, index_z] = pk_dm

        '''
        # Recover the non_linear scale computed by halofit. If no scale was
        # affected, set the scale to one, and make sure that the nuisance
        # parameter epsilon is set to zero
        if cosmo.nonlinear_method == 0:
            self.k_sigma[:] = 1.e6
        else:
            self.k_sigma = cosmo.nonlinear_scale(self.z_p, self.nzmax)

        # Define the alpha function, that will characterize the theoretical
        # uncertainty. Chosen to be 0.001 at low k, raise between 0.1 and 0.2
        # to self.theoretical_error
        if self.theoretical_error != 0:
            for index_l in xrange(self.nlmax):
                k = (self.l[index_l] + 0.5) / self.r
                self.alpha[index_l, :] = np.log(1. + k[:] / self.k_sigma[:]) / (1. + np.log(1. + k[:] / self.k_sigma[:])) * self.theoretical_error

        # recover the e_th_nu part of the error function
        e_th_nu = self.coefficient_f_nu * cosmo.Omega_nu / cosmo.Omega_m()

        # Compute the Error E_th_nu function
        if 'epsilon' in self.use_nuisance:
            for index_l in xrange(self.nlmax):
                self.E_th_nu[index_l, :] = np.log(1. + self.l[index_l] / self.k_sigma[:] * self.r[:]) / (1. + np.log(1. + self.l[index_l] / self.k_sigma[:] * self.r[:])) * e_th_nu

        # Add the error function, with the nuisance parameter, to P_nl_th, if
        # the nuisance parameter exists
        for index_l in xrange(self.nlmax):
            epsilon = data.mcmc_parameters['epsilon']['current'] * (data.mcmc_parameters['epsilon']['scale'])
            self.pk[index_l, :] *= (1. + epsilon * self.E_th_nu[index_l, :])
        '''

        Cl_GG_integrand = np.zeros_like(self.Cl_integrand)
        Cl_GG = np.zeros_like(self.Cl)

        if intrinsic_alignment:
            Cl_II_integrand = np.zeros_like(self.Cl_integrand)
            Cl_II = np.zeros_like(self.Cl)

            Cl_GI_integrand = np.zeros_like(self.Cl_integrand)
            Cl_GI = np.zeros_like(self.Cl)

        dr = self.r[1:] - self.r[:-1]
        # Start loop over l for computation of C_l^shear
        # Start loop over l for computation of E_l
        for il in xrange(self.nlmax):
            # find Cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
            for Bin1 in xrange(self.nzbins):
                for Bin2 in xrange(Bin1, self.nzbins):
                    Cl_GG_integrand[:, self.one_dim_index(Bin1,Bin2)] = self.g[:, Bin1] * self.g[:, Bin2] / self.r**2 * self.pk[il, :]
                    #print(self.Cl_integrand)
                    if intrinsic_alignment:
                        factor_IA = self.get_IA_factor(self.z_p, linear_growth_rate, amp_IA, exp_IA)
                        #print(self.eta_r[1:, zbin1].shape)
                        Cl_II_integrand[:, self.one_dim_index(Bin1,Bin2)] = self.pr[:, Bin1] * self.pr[:, Bin2] * factor_IA**2 / self.r**2 * self.pk[il, :]
                        Cl_GI_integrand[:, self.one_dim_index(Bin1,Bin2)] = (self.g[:, Bin1] * self.pr[:, Bin2] + self.g[:, Bin2] * self.pr[:, Bin1]) * factor_IA / self.r**2 * self.pk[il, :]

                    '''
                    if self.theoretical_error != 0:
                        self.El_integrand[1:, self.one_dim_index(Bin1, Bin2)] = self.g[:, Bin1] * self.g[:, Bin2] / self.r**2 * self.pk[il, :] * self.alpha[il, :]
                    '''

            # Integrate over r to get C_l^shear_ij = P_ij(l)
            # C_l^shear_ij = 9/16 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r)
            # g_j(r) /r**2) P(k=l/r,z(r)) dr
            # It is then multiplied by 9/16*Omega_m**2
            # and then by (h/2997.9)**4 to be dimensionless
            # (since P(k)*dr is in units of Mpc**4)
            for Bin in xrange(self.nzcorrs):
                Cl_GG[il, Bin] = np.sum(0.5 * (Cl_GG_integrand[1:, Bin] + Cl_GG_integrand[:-1, Bin]) * dr)
                Cl_GG[il, Bin] *= 9. / 16. * self.Omega_m**2
                Cl_GG[il, Bin] *= (self.small_h / 2997.9)**4

                if intrinsic_alignment:
                    Cl_II[il, Bin] = np.sum(0.5 * (Cl_II_integrand[1:, Bin] + Cl_II_integrand[:-1, Bin]) * dr)

                    Cl_GI[il, Bin] = np.sum(0.5 * (Cl_GI_integrand[1:, Bin] + Cl_GI_integrand[:-1, Bin]) * dr)
                    # here we divide by 4, because we get a 2 from g(r)!
                    Cl_GI[il, Bin] *= 3. / 4. * self.Omega_m
                    Cl_GI[il, Bin] *= (self.small_h / 2997.9)**2

                '''
                if self.theoretical_error != 0:
                    self.El[il, Bin] = np.sum(0.5 * (self.El_integrand[1:, Bin] + self.El_integrand[:-1, Bin]) * dr)
                    self.El[il, Bin] *= 9. / 16. * self.Omega_m**2
                    self.El[il, Bin] *= (self.small_h / 2997.9)**4
                '''

            '''
            for Bin1 in xrange(self.nzbins):
                Cl_GG[il, self.one_dim_index(Bin1, Bin1)] += self.noise
            '''
        if intrinsic_alignment:
            self.Cl = Cl_GG + Cl_GI + Cl_II
        else:
            self.Cl = Cl_GG
            #print(Cl_GG)
            #print(self.Cl)

        # Spline Cl[il,Bin1,Bin2] along l
        for Bin in xrange(self.nzcorrs):
            self.spline_Cl[Bin] = list(itp.splrep(self.l, self.Cl[:, Bin]))

        # Interpolate Cl at values lll and store results in Cll
        for Bin in xrange(self.nzcorrs):
            self.Cll[Bin,:] = itp.splev(self.lll[:], self.spline_Cl[Bin])

        # Start loop over theta values
        for it in xrange(self.nthetatot):
            ilmax = self.il_max[it]

            self.BBessel0[:ilmax] = special.j0(self.lll[:ilmax] * self.theta[it] * self.a2r)
            self.BBessel4[:ilmax] = special.jv(4, self.lll[:ilmax] * self.theta[it] * self.a2r)

            # Here is the actual trapezoidal integral giving the xi's:
            # - in more explicit style:
            # for Bin in xrange(self.nzcorrs):
            #     for il in xrange(ilmax):
            #         self.xi1[it, Bin] = np.sum(self.ldl[il]*self.Cll[Bin,il]*self.BBessel0[il])
            #         self.xi2[it, Bin] = np.sum(self.ldl[il]*self.Cll[Bin,il]*self.BBessel4[il])
            # - in more compact and vectorizable style:
            self.xi1[it, :] = np.sum(self.ldl[:ilmax] * self.Cll[:, :ilmax] * self.BBessel0[:ilmax], axis=1)
            self.xi2[it, :] = np.sum(self.ldl[:ilmax] * self.Cll[:, :ilmax] * self.BBessel4[:ilmax], axis=1)

        # normalisation of xi's
        self.xi1 = self.xi1 / (2. * math.pi)
        self.xi2 = self.xi2 / (2. * math.pi)

        # Spline the xi's
        for Bin in xrange(self.nzcorrs):
            self.xi1_theta[Bin] = list(itp.splrep(self.theta, self.xi1[:,Bin]))
            self.xi2_theta[Bin] = list(itp.splrep(self.theta, self.xi2[:,Bin]))

        # Get xi's in same column vector format as the data
        #iz = 0
        #for Bin in xrange(self.nzcorrs):
        #    iz = iz + 1  # this counts the bin combinations
        #    for i in xrange(self.ntheta):
        #        j = (iz-1)*2*self.ntheta
        #        self.xi[j+i] = itp.splev(
        #            self.theta_bins[i], self.xi1_theta[Bin])
        #        self.xi[self.ntheta + j+i] = itp.splev(
        #            self.theta_bins[i], self.xi2_theta[Bin])
        # or in more compact/vectorizable form:
        iz = 0
        for Bin in xrange(self.nzcorrs):
            iz = iz + 1  # this counts the bin combinations
            j = (iz - 1) * 2 * self.ntheta
            self.xi[j:j + self.ntheta] = itp.splev(self.theta_bins[:self.ntheta], self.xi1_theta[Bin])
            self.xi[j + self.ntheta:j + 2 * self.ntheta] = itp.splev(self.theta_bins[:self.ntheta], self.xi2_theta[Bin])

        # final chi2
        vec = self.xi[self.mask_indices] - self.xi_obs[self.mask_indices]
        #print(self.xi_obs[self.mask_indices], len(self.xi_obs[self.mask_indices]))
        #print(self.xi[self.mask_indices], len(self.xi[self.mask_indices]))

        # this is for running smoothly with MultiNest
        # (in initial checking of prior space, there might occur weird solutions)
        if np.isinf(vec).any() or np.isnan(vec).any():
            chi2 = 2e12
        else:
            # don't invert that matrix...
            # use the Cholesky decomposition instead:
            yt = solve_triangular(self.cholesky_transform, vec, lower=True)
            chi2 = yt.dot(yt)

        return -chi2 / 2.

    #######################################################################################################
    # This function is used to convert 2D sums over the two indices (Bin1, Bin2) of an N*N symmetric matrix
    # into 1D sums over one index with N(N+1)/2 possible values
    def one_dim_index(self, Bin1, Bin2):
        if Bin1 <= Bin2:
            return Bin2 + self.nzbins * Bin1 - (Bin1 * (Bin1 + 1)) // 2
        else:
            return Bin1 + self.nzbins * Bin2 - (Bin2 * (Bin2 + 1)) // 2
