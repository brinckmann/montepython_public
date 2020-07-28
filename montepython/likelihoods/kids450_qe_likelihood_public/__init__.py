#!/usr/bin/env python
# encoding: UTF8
#
########################################################################
# likelihood for the KiDS-450 WL power spectrum (quadratic estimator)  #
########################################################################
#
# Developed by F. Koehlinger based on and by largely adapting
# Benjamin Audren's Monte Python likelihood euclid_lensing
#
# To be used with data from F. Koehlinger et al. 2017 (MNRAS, 471, 4412;
# arXiv:1706.02892) which can be downloaded from:
#
# http://kids.strw.leidenuniv.nl/sciencedata.php
#
########################################################################

from montepython.likelihood_class import Likelihood
import io_mp
import parser_mp
import os
import numpy as np
import scipy.interpolate as itp
from scipy.linalg import cholesky, solve_triangular
import time

try:
    xrange
except NameError:
    xrange = range

class kids450_qe_likelihood_public(Likelihood):

    def __init__(self, path, data, command_line):
        # I should already take care of using only GRF mocks or data here (because of different folder-structures etc...)
        # or for now just write it for GRFs for tests and worry about it later...
        Likelihood.__init__(self, path, data, command_line)

        # Check if the data can be found
        print("Trying to read %s", self.data_directory+'Resetting_bias/parameters_B_mode_model.dat')
        try:
            fname = os.path.join(self.data_directory, 'Resetting_bias/parameters_B_mode_model.dat')
            parser_mp.existing_file(fname)
        except:
            raise io_mp.ConfigurationError('KiDS-450 QE data not found. Download the data at '
                                           'http://kids.strw.leidenuniv.nl/sciencedata.php '
                                           'and specify path to data through the variable '
                                           'kids450_qe_likelihood_public.data_directory in '
                                           'the .data file. See README in likelihood folder '
                                           'for further instructions.')

        # TODO: this is also CFHTLenS legacy...
        # only relevant for GRFs!
        #dict_BWM = {'W1': 'G10_', 'W2': 'G126_', 'W3': 'G162_', 'W4': 'G84_'}

        self.need_cosmo_arguments(data, {'output':'mPk'})

        self.redshift_bins = []
        for index_zbin in xrange(len(self.zbin_min)):
            redshift_bin = '{:.2f}z{:.2f}'.format(self.zbin_min[index_zbin], self.zbin_max[index_zbin])
            self.redshift_bins.append(redshift_bin)

        # number of z-bins
        self.nzbins = len(self.redshift_bins)
        # number of *unique* correlations between z-bins
        self.nzcorrs = self.nzbins * (self.nzbins + 1) // 2

        all_bands_EE_to_use = []
        all_bands_BB_to_use = []

        '''
        if self.fit_cross_correlations_only:
            # mask out auto-spectra:
            for index_zbin1 in xrange(self.nzbins):
                for index_zbin2 in xrange(index_zbin1 + 1):
                    if index_zbin1 == index_zbin2:
                        all_bands_EE_to_use += np.zeros_like(self.bands_EE_to_use).tolist()
                        all_bands_BB_to_use += np.zeros_like(self.bands_BB_to_use).tolist()
                    else:
                        all_bands_EE_to_use += self.bands_EE_to_use
                        all_bands_BB_to_use += self.bands_BB_to_use

        else:
            # default, use all correlations:
            for i in xrange(self.nzcorrs):
                all_bands_EE_to_use += self.bands_EE_to_use
                all_bands_BB_to_use += self.bands_BB_to_use
        '''
        # default, use all correlations:
        for i in xrange(self.nzcorrs):
            all_bands_EE_to_use += self.bands_EE_to_use
            all_bands_BB_to_use += self.bands_BB_to_use

        all_bands_to_use = np.concatenate((all_bands_EE_to_use, all_bands_BB_to_use))
        self.indices_for_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]

        # this is also the number of points in the datavector
        ndata = len(self.indices_for_bands_to_use)

        # I should load all the data needed only once, i.e. HERE:
        # not so sure about statement above, I have the feeling "init" is called for every MCMC step...
        # maybe that's why the memory is filling up on other machines?! --> nope, that wasn't the reason...
        start_load = time.time()

        if self.correct_resetting_bias:
            fname = os.path.join(self.data_directory, 'Resetting_bias/parameters_B_mode_model.dat')
            A_B_modes, exp_B_modes, err_A_B_modes, err_exp_B_modes = np.loadtxt(fname, unpack=True)
            self.params_resetting_bias = np.array([A_B_modes, exp_B_modes])
            fname = os.path.join(self.data_directory, 'Resetting_bias/covariance_B_mode_model.dat')
            self.cov_resetting_bias = np.loadtxt(fname)

        # try to load fiducial m-corrections from file (currently these are global values over full field, hence no looping over fields required for that!)
        # TODO: Make output dependent on field, not necessary for current KiDS approach though!
        try:
            fname = os.path.join(self.data_directory, '{:}zbins/m_correction_avg.txt'.format(self.nzbins))
            if self.nzbins == 1:
                self.m_corr_fiducial_per_zbin = np.asarray([np.loadtxt(fname, usecols=[1])])
            else:
                self.m_corr_fiducial_per_zbin = np.loadtxt(fname, usecols=[1])
        except:
            self.m_corr_fiducial_per_zbin = np.zeros(self.nzbins)
            print('Could not load m-correction values from \n', fname)
            print('Setting them to zero instead.')

        try:
            fname = os.path.join(self.data_directory, '{:}zbins/sigma_int_n_eff_{:}zbins.dat'.format(self.nzbins, self.nzbins))
            tbdata = np.loadtxt(fname)
            if self.nzbins == 1:
                # correct columns for file!
                sigma_e1 = np.asarray([tbdata[2]])
                sigma_e2 = np.asarray([tbdata[3]])
                n_eff = np.asarray([tbdata[4]])
            else:
                # correct columns for file!
                sigma_e1 = tbdata[:, 2]
                sigma_e2 = tbdata[:, 3]
                n_eff = tbdata[:, 4]

            self.sigma_e = np.sqrt((sigma_e1**2 + sigma_e2**2) / 2.)
            # convert from 1 / sq. arcmin to 1 / sterad
            self.n_eff = n_eff / np.deg2rad(1. / 60.)**2
        except:
            # these dummies will set noise power always to 0!
            self.sigma_e = np.zeros(self.nzbins)
            self.n_eff = np.ones(self.nzbins)
            print('Could not load sigma_e and n_eff!')

        collect_bp_EE_in_zbins = []
        collect_bp_BB_in_zbins = []
        # collect BP per zbin and combine into one array
        for zbin1 in xrange(self.nzbins):
            for zbin2 in xrange(zbin1 + 1): #self.nzbins):
                # zbin2 first in fname!
                fname_EE = os.path.join(self.data_directory, '{:}zbins/band_powers_EE_z{:}xz{:}.dat'.format(self.nzbins, zbin1 + 1, zbin2 + 1))
                fname_BB = os.path.join(self.data_directory, '{:}zbins/band_powers_BB_z{:}xz{:}.dat'.format(self.nzbins, zbin1 + 1, zbin2 + 1))
                extracted_band_powers_EE = np.loadtxt(fname_EE)
                extracted_band_powers_BB = np.loadtxt(fname_BB)
                collect_bp_EE_in_zbins.append(extracted_band_powers_EE)
                collect_bp_BB_in_zbins.append(extracted_band_powers_BB)

        self.band_powers = np.concatenate((np.asarray(collect_bp_EE_in_zbins).flatten(), np.asarray(collect_bp_BB_in_zbins).flatten()))

        fname = os.path.join(self.data_directory, '{:}zbins/covariance_all_z_EE_BB.dat'.format(self.nzbins))
        self.covariance = np.loadtxt(fname)

        fname = os.path.join(self.data_directory, '{:}zbins/band_window_matrix_nell100.dat'.format(self.nzbins))
        self.band_window_matrix = np.loadtxt(fname)
        # ells_intp and also band_offset are consistent between different patches!

        fname = os.path.join(self.data_directory, '{:}zbins/multipole_nodes_for_band_window_functions_nell100.dat'.format(self.nzbins))
        self.ells_intp = np.loadtxt(fname)
        self.band_offset_EE = len(extracted_band_powers_EE)
        self.band_offset_BB = len(extracted_band_powers_BB)

        # Check if any of the n(z) needs to be shifted in loglkl by D_z{1...n}:
        self.shift_n_z_by_D_z = np.zeros(self.nzbins, 'bool')
        for zbin in xrange(self.nzbins):
            param_name = 'D_z{:}'.format(zbin + 1)
            if param_name in data.mcmc_parameters:
                self.shift_n_z_by_D_z[zbin] = True

        # Read fiducial dn_dz from window files:
        # TODO: the hardcoded z_min and z_max correspond to the lower and upper
        # endpoints of the shifted left-border histogram!
        z_samples = []
        hist_samples = []
        for zbin in xrange(self.nzbins):
            redshift_bin = self.redshift_bins[zbin]
            window_file_path = os.path.join(
                self.data_directory, '{:}/n_z_avg_{:}.hist'.format(self.photoz_method, redshift_bin))
            if os.path.exists(window_file_path):
                zptemp, hist_pz = np.loadtxt(window_file_path, usecols=[0, 1], unpack=True)
                shift_to_midpoint = np.diff(zptemp)[0] / 2.
                if zbin > 0:
                    zpcheck = zptemp
                    if np.sum((zptemp - zpcheck)**2) > 1e-6:
                        raise io_mp.LikelihoodError('The redshift values for the window files at different bins do not match.')
                print('Loaded n(zbin{:}) from: \n'.format(zbin + 1), window_file_path)
                # we add a zero as first element because we want to integrate down to z = 0!
                z_samples += [np.concatenate((np.zeros(1), zptemp + shift_to_midpoint))]
                hist_samples += [np.concatenate((np.zeros(1), hist_pz))]
            else:
                raise io_mp.LikelihoodError("File not found:\n %s"%window_file_path)

        z_samples = np.asarray(z_samples)
        hist_samples = np.asarray(hist_samples)

        # prevent undersampling of histograms!
        if self.nzmax < len(zptemp):
            print("You're trying to integrate at lower resolution than supplied by the n(z) histograms. \n Increase nzmax! Aborting now...")
            exit()
        # if that's the case, we want to integrate at histogram resolution and need to account for
        # the extra zero entry added
        elif self.nzmax == len(zptemp):
            self.nzmax = z_samples.shape[1]
            # requires that z-spacing is always the same for all bins...
            self.redshifts = z_samples[0, :]
            print('Integrations performed at resolution of histogram!')
        # if we interpolate anyway at arbitrary resolution the extra 0 doesn't matter
        else:
            self.nzmax += 1
            self.redshifts = np.linspace(z_samples.min(), z_samples.max(), self.nzmax)
            print('Integration performed at set nzmax resolution!')

        self.pz = np.zeros((self.nzmax, self.nzbins))
        self.pz_norm = np.zeros(self.nzbins, 'float64')
        for zbin in xrange(self.nzbins):
                # we assume that the histograms loaded are given as left-border histograms
                # and that the z-spacing is the same for each histogram
                spline_pz = itp.splrep(z_samples[zbin, :], hist_samples[zbin, :])

                #z_mod = self.z_p
                mask_min = self.redshifts >= z_samples[zbin, :].min()
                mask_max = self.redshifts <= z_samples[zbin, :].max()
                mask = mask_min & mask_max
                # points outside the z-range of the histograms are set to 0!
                self.pz[mask, zbin] = itp.splev(self.redshifts[mask], spline_pz)
                # Normalize selection functions
                dz = self.redshifts[1:] - self.redshifts[:-1]
                self.pz_norm[zbin] = np.sum(0.5 * (self.pz[1:, zbin] + self.pz[:-1, zbin]) * dz)

        self.z_max = self.redshifts.max()

        # k_max is arbitrary at the moment, since cosmology module is not calculated yet...TODO
        if self.mode == 'halofit':
            self.need_cosmo_arguments(data, {'z_max_pk': self.z_max, 'output': 'mPk', 'non linear': self.mode, 'P_k_max_h/Mpc': self.k_max_h_by_Mpc})
        else:
            self.need_cosmo_arguments(data, {'z_max_pk': self.z_max, 'output': 'mPk', 'P_k_max_h/Mpc': self.k_max_h_by_Mpc})

        print('Time for loading all data files:', time.time() - start_load)

        fname = os.path.join(self.data_directory, 'number_datapoints.txt')
        np.savetxt(fname, [ndata], header='number of datapoints in masked datavector')

        return

    def loglkl(self, cosmo, data):

        # class' Omega_m includes Omega_nu etc!
        # but Omega0_m doesn't!
        # ATTENTION: definition of Omega_m in CLASS has changed again: Omega_m = self.ba.Omega0_cdm+self.ba.Omega0_b
        # But I think Omega_m should also contain densities of other species!!!
        #Omega_m = cosmo.Omega_m()
        # this is now a copy of what is returned as Omega_m to MontePython:
        # that didn't work, because ".ba" is not available...
        #Omega_m = cosmo.ba.Omega0_b + cosmo.ba.Omega0_cdm + cosmo.ba.Omega0_ncdm_tot + cosmo.ba.Omega0_dcdm
        # Next try:
        # Omega_m() = self.ba.Omega0_cdm+self.ba.Omega0_b
        # Omega_nu = self.ba.Omega0_ncdm_tot
        # only contributions from decaying DM missing...
        # be careful though, if at some point Omega_m is defined again to contain every species' contribution
        # it does contain all species again in CLASS 2.5.0! #+ cosmo.Omega_nu
        # TODO: Always check definition of cosmo.Omega_m() in classy.pyx!!!
        self.Omega_m = cosmo.Omega_m()
        self.small_h = cosmo.h()

        # m-correction:
        # Errors on m-corrections for different z-bins are correlated, thus one free nuisance "m_corr" is enough,
        # We fix the amplitude to the 2\sigma range around the fiducial m-correction value from the lowest redshift-bin
        # for that and add the delta_m to all fiducial m-corrections, hence:
        param_name = 'm_corr'
        if param_name in data.mcmc_parameters:
            m_corr = data.mcmc_parameters[param_name]['current'] * data.mcmc_parameters[param_name]['scale']
            #ATTENTION: sign matters and this order is the correct one for correlation if delta_m_corr is added!
            delta_m_corr = m_corr - self.m_corr_fiducial_per_zbin[0]
            # this is wrong!
            #m_corr_per_zbin = [m_corr_z1]
            #m_corr_per_zbin = [self.m_corr_fiducial_per_zbin[0] + delta_m_corr]
            m_corr_per_zbin = np.zeros(self.nzbins)
            for zbin in xrange(0, self.nzbins):
                m_corr_per_zbin[zbin] = self.m_corr_fiducial_per_zbin[zbin] + delta_m_corr
        else:
            # if "m_corr" is not specified in input parameter script we just apply the fiducial m-correction values
            # if these could not be loaded, this vector contains only zeros!
            m_corr_per_zbin = self.m_corr_fiducial_per_zbin

        # draw m-correction now instead from a multivariate Gaussian taking the fully correlated errors into account:
        # this does not yield converging chains in reasonable runtimes (e.g. 3 z-bins > 1000 CPUh...)
        '''
        if self.marginalize_over_multiplicative_bias:
            if self.nzbins > 1:
                m_corr_per_zbin = np.random.multivariate_normal(self.m_corr_fiducial_per_zbin, self.cov_m_corr)
                #print('m-correction')
                #print(self.m_corr_fiducial_per_zbin, self.cov_m_corr)
                #print(m_corr_per_zbin)
            else:
                m_corr_per_zbin = np.random.normal(self.m_corr_fiducial_per_zbin, self.err_multiplicative_bias)
        else:
            m_corr_per_zbin = self.m_corr_fiducial_per_zbin
        '''

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

        if intrinsic_alignment:
            self.rho_crit = self.get_critical_density()
            # derive the linear growth factor D(z)
            linear_growth_rate = np.zeros_like(self.redshifts)
            #print(self.redshifts)
            for index_z, z in enumerate(self.redshifts):
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

        #residual noise correction amplitude:
        #param_name = 'A_noise'
        # zeros == False!
        A_noise = np.zeros(self.nzbins)
        add_noise_power = np.zeros(self.nzbins, dtype=bool)
        param_name = 'A_noise_corr'
        if param_name in data.mcmc_parameters:
            # assume correlated apmlitudes for the noise-power (i.e. same amplitude for all autocorrelations):
            A_noise[:] = data.mcmc_parameters[param_name]['current'] * data.mcmc_parameters[param_name]['scale']
            add_noise_power[:] = True
        else:
            # assume uncorrelated amplitudes for the noise-power:
            for zbin in xrange(self.nzbins):

                param_name = 'A_noise_z{:}'.format(zbin + 1)

                if param_name in data.mcmc_parameters:
                    A_noise[zbin] = data.mcmc_parameters[param_name]['current'] * data.mcmc_parameters[param_name]['scale']
                    add_noise_power[zbin] = True

        # this is not correct, if this is considered to be a calibration!
        '''
        # this is all for B-mode power-law model:
        param_name1 = 'A_B_modes'
        param_name2 = 'exp_B_modes'
        use_B_mode_model = False
        if param_name1 in data.mcmc_parameters and param_name2 in data.mcmc_parameters:
            amp_BB = data.mcmc_parameters[param_name1]['current'] * data.mcmc_parameters[param_name1]['scale']
            exp_BB = data.mcmc_parameters[param_name2]['current'] * data.mcmc_parameters[param_name2]['scale']
            use_B_mode_model = True
        '''
        # this was the fiducial approach for the first submission
        # the one above might be faster (and more consistent)
        if self.correct_resetting_bias:
            #A_B_modes = np.random.normal(self.best_fit_A_B_modes, self.best_fit_err_A_B_modes)
            #exp_B_modes = np.random.normal(self.best_fit_exp_B_modes, self.best_fit_err_exp_B_modes)
            amp_BB, exp_BB = np.random.multivariate_normal(self.params_resetting_bias, self.cov_resetting_bias)
            #print('resetting bias')
            #print(self.params_resetting_bias, self.cov_resetting_bias)
            #print(amp_BB, exp_BB)

        # get distances from cosmo-module:
        r, dzdr = cosmo.z_of_r(self.redshifts)

        # 1) determine l-range for taking the sum, #l = l_high-l_min at least!!!:
        # this is the correct calculation!
        # for real data, I should start sum from physical scales, i.e., currently l>= 80!
        # TODO: Set this automatically!!! --> not automatically yet, but controllable via "myCFHTLenS_tomography.data"!!!
        # these are integer l-values over which we will take the sum used in the convolution with the band window matrix
        ells_min = self.ells_intp[0]
        '''
        if self.key == 'data_XinPi':
            ells_sum = self.ell_bin_centers
            # TODO: This might cause trouble!!!
            ells_max = 5150.
        else:
            ells_max = self.ells_intp[-1]
            nells = int(ells_max - ells_min + 1)
            ells_sum = np.linspace(ells_min, ells_max, nells)
        '''
        ells_max = self.ells_intp[-1]
        nells = int(ells_max - ells_min + 1)
        ells_sum = np.linspace(ells_min, ells_max, nells)

        # these are the l-nodes for the derivation of the theoretical Cl:
        ells = np.logspace(np.log10(ells_min), np.log10(ells_max), self.nellsmax)

        # After long and extensive testing:
        # Don't put calls to Class (i.e. cosmo...) into a loop...
        # before "pk" and the constants were just called at demand below in the code (due to convenience an copy & paste)
        # which seemed to have been the source for the memory leak...

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        # this doesn't really have to go into the loop over fields!
        pk = np.zeros((self.nellsmax, self.nzmax),'float64')
        k_max_in_inv_Mpc = self.k_max_h_by_Mpc * self.small_h
        for index_ells in xrange(self.nellsmax):
            for index_z in xrange(1, self.nzmax):
                # standard Limber approximation:
                #k = ells[index_ells] / r[index_z]
                # extended Limber approximation (cf. LoVerde & Afshordi 2008):
                k_in_inv_Mpc = (ells[index_ells] + 0.5) / r[index_z]
                if k_in_inv_Mpc > k_max_in_inv_Mpc:
                    pk_dm = 0.
                else:
                    pk_dm = cosmo.pk(k_in_inv_Mpc, self.redshifts[index_z])
                #pk[index_ells,index_z] = cosmo.pk(ells[index_ells]/r[index_z], self.redshifts[index_z])
                if self.baryon_feedback:
                    if 'A_bary' in data.mcmc_parameters:
                        A_bary = data.mcmc_parameters['A_bary']['current'] * data.mcmc_parameters['A_bary']['scale']
                        #print('A_bary={:.4f}'.format(A_bary))
                        pk[index_ells, index_z] = pk_dm * self.baryon_feedback_bias_sqr(k_in_inv_Mpc / self.small_h, self.redshifts[index_z], A_bary=A_bary)
                    else:
                        pk[index_ells, index_z] = pk_dm * self.baryon_feedback_bias_sqr(k_in_inv_Mpc / self.small_h, self.redshifts[index_z])
                else:
                    pk[index_ells, index_z] = pk_dm

        # for KiDS-450 constant biases in photo-z are not sufficient:
        if self.bootstrap_photoz_errors:
            # draw a random bootstrap n(z); borders are inclusive!
            random_index_bootstrap = np.random.randint(int(self.index_bootstrap_low), int(self.index_bootstrap_high) + 1)
            #print('Bootstrap index:', random_index_bootstrap)
            pz = np.zeros((self.nzmax, self.nzbins), 'float64')
            pz_norm = np.zeros(self.nzbins, 'float64')
            for zbin in xrange(self.nzbins):

                redshift_bin = self.redshift_bins[zbin]
                #ATTENTION: hard-coded subfolder!
                #index can be recycled since bootstraps for tomographic bins are independent!
                fname = os.path.join(self.data_directory, '{:}/bootstraps/{:}/n_z_avg_bootstrap{:}.hist'.format(self.photoz_method, redshift_bin, random_index_bootstrap))
                z_hist, n_z_hist = np.loadtxt(fname, unpack=True)

                param_name = 'D_z{:}'.format(zbin + 1)
                if param_name in data.mcmc_parameters:
                    z_mod = self.redshifts + data.mcmc_parameters[param_name]['current'] * data.mcmc_parameters[param_name]['scale']
                else:
                    z_mod = self.redshifts

                shift_to_midpoint = np.diff(z_hist)[0] / 2.
                spline_pz = itp.splrep(z_hist + shift_to_midpoint, n_z_hist)
                mask_min = z_mod >= z_hist.min() + shift_to_midpoint
                mask_max = z_mod <= z_hist.max() + shift_to_midpoint
                mask = mask_min & mask_max
                # points outside the z-range of the histograms are set to 0!
                pz[mask, zbin] = itp.splev(z_mod[mask], spline_pz)

                dz = self.redshifts[1:] - self.redshifts[:-1]
                pz_norm[zbin] = np.sum(0.5 * (pz[1:, zbin] + pz[:-1, zbin]) * dz)

            pr = pz * (dzdr[:, np.newaxis] / pz_norm)

        elif (not self.bootstrap_photoz_errors) and (self.shift_n_z_by_D_z.any()):

            pz = np.zeros((self.nzmax, self.nzbins), 'float64')
            pz_norm = np.zeros(self.nzbins, 'float64')
            for zbin in xrange(self.nzbins):

                param_name = 'D_z{:}'.format(zbin + 1)
                if param_name in data.mcmc_parameters:
                    z_mod = self.redshifts + data.mcmc_parameters[param_name]['current'] * data.mcmc_parameters[param_name]['scale']
                else:
                    z_mod = self.redshifts
                # Load n(z) again:
                redshift_bin = self.redshift_bins[zbin]
                fname = os.path.join(self.data_directory, '{:}/n_z_avg_{:}.hist'.format(self.photoz_method, redshift_bin))
                z_hist, n_z_hist = np.loadtxt(fname, usecols=(0, 1), unpack=True)
                shift_to_midpoint = np.diff(z_hist)[0] / 2.
                spline_pz = itp.splrep(z_hist + shift_to_midpoint, n_z_hist)
                mask_min = z_mod >= z_hist.min() + shift_to_midpoint
                mask_max = z_mod <= z_hist.max() + shift_to_midpoint
                mask = mask_min & mask_max
                # points outside the z-range of the histograms are set to 0!
                pz[mask, zbin] = itp.splev(z_mod[mask], spline_pz)
                # Normalize selection functions
                dz = self.redshifts[1:] - self.redshifts[:-1]
                pz_norm[zbin] = np.sum(0.5 * (pz[1:, zbin] + pz[:-1, zbin]) * dz)

            pr = pz * (dzdr[:, np.newaxis] / pz_norm)

        else:
            pr = self.pz * (dzdr[:, np.newaxis] / self.pz_norm)

        # Compute function g_i(r), that depends on r and the bin
        # g_i(r) = 2r(1+z(r)) int_r^+\infty drs eta_r(rs) (rs-r)/rs
        g = np.zeros((self.nzmax, self.nzbins), 'float64')
        for zbin in xrange(self.nzbins):
            # assumes that z[0] = 0
            for nr in xrange(1, self.nzmax - 1):
            #for nr in xrange(self.nzmax - 1):
                fun = pr[nr:, zbin] * (r[nr:] - r[nr]) / r[nr:]
                g[nr, zbin] = np.sum(0.5 * (fun[1:] + fun[:-1]) * (r[nr + 1:] - r[nr:-1]))
                g[nr, zbin] *= 2. * r[nr] * (1. + self.redshifts[nr])

        # Start loop over l for computation of C_l^shear
        Cl_GG_integrand = np.zeros((self.nzmax, self.nzbins, self.nzbins), 'float64')
        Cl_GG = np.zeros((self.nellsmax, self.nzbins, self.nzbins), 'float64')

        if intrinsic_alignment:
            Cl_II_integrand = np.zeros_like(Cl_GG_integrand)
            Cl_II = np.zeros_like(Cl_GG)

            Cl_GI_integrand = np.zeros_like(Cl_GG_integrand)
            Cl_GI = np.zeros_like(Cl_GG)

        dr = r[1:] - r[:-1]
        # removing shifts like array[1:, ...] which assume that z[0] = 0:
        for index_ell in xrange(self.nellsmax):

            # find Cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
            for zbin1 in xrange(self.nzbins):
                for zbin2 in xrange(zbin1 + 1): #self.nzbins):
                    Cl_GG_integrand[1:, zbin1, zbin2] = g[1:, zbin1] * g[1:, zbin2] / r[1:]**2 * pk[index_ell, 1:]

                    if intrinsic_alignment:
                        factor_IA = self.get_factor_IA(self.redshifts[1:], linear_growth_rate[1:], amp_IA, exp_IA) #/ self.dzdr[1:]
                        #print(F_of_x)
                        #print(self.eta_r[1:, zbin1].shape)
                        Cl_II_integrand[1:, zbin1, zbin2] = pr[1:, zbin1] * pr[1:, zbin2] * factor_IA**2 / r[1:]**2 * pk[index_ell, 1:]
                        Cl_GI_integrand[1:, zbin1, zbin2] = (g[1:, zbin1] * pr[1:, zbin2] + g[1:, zbin2] * pr[1:, zbin1]) * factor_IA / r[1:]**2 * pk[index_ell, 1:]

            # Integrate over r to get C_l^shear_ij = P_ij(l)
            # C_l^shear_ii = 9/4 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r) g_j(r) /r**2) P(k=l/r,z(r))
            for zbin1 in xrange(self.nzbins):
                for zbin2 in xrange(zbin1 + 1): #self.nzbins):
                    Cl_GG[index_ell, zbin1, zbin2] = np.sum(0.5 * (Cl_GG_integrand[1:, zbin1, zbin2] + Cl_GG_integrand[:-1, zbin1, zbin2]) * dr)
                    # here we divide by 16, because we get a 2^2 from g(z)!
                    Cl_GG[index_ell, zbin1, zbin2] *= 9. / 16. * self.Omega_m**2 # in units of Mpc**4
                    Cl_GG[index_ell, zbin1, zbin2] *= (self.small_h / 2997.9)**4 # dimensionless

                    if intrinsic_alignment:
                        Cl_II[index_ell, zbin1, zbin2] = np.sum(0.5 * (Cl_II_integrand[1:, zbin1, zbin2] + Cl_II_integrand[:-1, zbin1, zbin2]) * dr)

                        Cl_GI[index_ell, zbin1, zbin2] = np.sum(0.5 * (Cl_GI_integrand[1:, zbin1, zbin2] + Cl_GI_integrand[:-1, zbin1, zbin2]) * dr)
                        # here we divide by 4, because we get a 2 from g(r)!
                        Cl_GI[index_ell, zbin1, zbin2] *= 3. / 4. * self.Omega_m
                        Cl_GI[index_ell, zbin1, zbin2] *= (self.small_h / 2997.9)**2

        if intrinsic_alignment:
            Cl = Cl_GG + Cl_GI + Cl_II
        else:
            Cl = Cl_GG

        # ordering of redshift bins is correct in definition of theory below!
        theory_EE = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')
        theory_BB = np.zeros((self.nzcorrs, self.band_offset_BB), 'float64')
        theory_noise_EE = np.zeros((self.nzcorrs, self.band_offset_EE), 'float64')
        theory_noise_BB = np.zeros((self.nzcorrs, self.band_offset_BB), 'float64')
        #print(theory.shape)
        index_corr = 0
        #A_noise_corr = np.zeros(self.nzcorrs)
        for zbin1 in xrange(self.nzbins):
            for zbin2 in xrange(zbin1 + 1): #self.nzbins):
                #correlation = 'z{:}z{:}'.format(zbin1 + 1, zbin2 + 1)
                ell_norm = ells_sum * (ells_sum + 1) / (2. * np.pi)
                # calculate m-correction vector here:
                # this loop goes over bands per z-corr; m-correction is the same for all bands in one tomographic bin!!!
                val_m_corr_EE = (1. + m_corr_per_zbin[zbin1]) * (1. + m_corr_per_zbin[zbin2]) * np.ones(len(self.bands_EE_to_use))
                val_m_corr_BB = (1. + m_corr_per_zbin[zbin1]) * (1. + m_corr_per_zbin[zbin2]) * np.ones(len(self.bands_BB_to_use))
                '''
                arg_a = (1. + A_noise[zbin1])
                arg_b = (1. + A_noise[zbin2])
                if np.sign(arg_a) < 0 and np.sign(arg_b) < 0:
                    sign = -1.
                elif np.sign(arg_a) < 0 or np.sign(arg_b) < 0:
                    sign = -1.
                else:
                    sign = 1.
                A_noise_corr[index_corr] = sign * self.sigma_e[zbin1] * self.sigma_e[zbin2] * np.sqrt(np.abs(arg_a)) * np.sqrt(np.abs(arg_b)) / (np.sqrt(self.n_eff[zbin1]) * np.sqrt(self.n_eff[zbin2]))
                '''
                # alternative definition, makes more sense than the one above:
                # I should add noise only to auto-correlations!
                if zbin1 == zbin2:
                    #A_noise_corr = self.sigma_e[zbin1] * self.sigma_e[zbin2] * (1. + A_noise[zbin1] + A_noise[zbin2]) / (np.sqrt(self.n_eff[zbin1]) * np.sqrt(self.n_eff[zbin2]))
                    # now the very simple definition should be sufficient!
                    A_noise_corr = A_noise[zbin1] * self.sigma_e[zbin1]**2 / self.n_eff[zbin1]
                else:
                    A_noise_corr = 0.
                Cl_sample = Cl[:, zbin1, zbin2]
                spline_Cl = itp.splrep(ells, Cl_sample)
                D_l_EE = ell_norm * itp.splev(ells_sum, spline_Cl)
                # TODO: 1e-9 can either become an adjustable constant or a parameter!
                # taking out ell_norm now (a constant times ell_norm is just another noise-power component)
                if self.correct_resetting_bias:
                    # TODO: get ell_centers...
                    #x_BB = ell_center * (ell_center + 1.) / (2. * np.pi) * self.sigma_e[zbin1] * self.sigma_e[zbin2] / np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])
                    # try to pull the model through the BWM first, that's more consistent with the code and doesn't require
                    x_BB = ell_norm * self.sigma_e[zbin1] * self.sigma_e[zbin2] / np.sqrt(self.n_eff[zbin1] * self.n_eff[zbin2])
                    D_l_BB = self.get_B_mode_model(x_BB, amp_BB, exp_BB)
                #else:
                #    D_l_BB = self.scale_B_modes # * ell_norm
                D_l_noise = ell_norm * A_noise_corr

                #theory[zbin1, zbin2, :] = get_theory(ells_sum, D_l, self.ells_intp, band_window_matrix, self.band_offset, correlation, bwm_style=self.bwm_style)
                '''
                if self.key == 'data_XinPi':
                    theory_EE[index_corr, :] = D_l_EE
                    theory_BB[index_corr, :] = 0.

                    if add_noise_power.all():
                        theory_noise_EE[index_corr, :] = D_l_noise
                        theory_noise_BB[index_corr, :] = 0.
                else:
                    theory_EE[index_corr, :] = self.get_theory(ells_sum, D_l_EE, self.band_window_matrix, index_corr, band_type_is_EE=True)
                    if self.correct_resetting_bias:
                        theory_BB[index_corr, :] = self.get_theory(ells_sum, D_l_BB, self.band_window_matrix, index_corr, band_type_is_EE=False)
                    else:
                        theory_BB[index_corr, :] = 0.

                    if add_noise_power.all():
                        theory_noise_EE[index_corr, :] = self.get_theory(ells_sum, D_l_noise, self.band_window_matrix, index_corr, band_type_is_EE=True)
                        theory_noise_BB[index_corr, :] = self.get_theory(ells_sum, D_l_noise, self.band_window_matrix, index_corr, band_type_is_EE=False)
                '''
                theory_EE[index_corr, :] = self.get_theory(ells_sum, D_l_EE, self.band_window_matrix, index_corr, band_type_is_EE=True)
                if self.correct_resetting_bias:
                    theory_BB[index_corr, :] = self.get_theory(ells_sum, D_l_BB, self.band_window_matrix, index_corr, band_type_is_EE=False)
                else:
                    theory_BB[index_corr, :] = 0.

                if add_noise_power.all():
                    theory_noise_EE[index_corr, :] = self.get_theory(ells_sum, D_l_noise, self.band_window_matrix, index_corr, band_type_is_EE=True)
                    theory_noise_BB[index_corr, :] = self.get_theory(ells_sum, D_l_noise, self.band_window_matrix, index_corr, band_type_is_EE=False)

                if index_corr == 0:
                    m_corr_EE = val_m_corr_EE
                    m_corr_BB = val_m_corr_BB
                else:
                    m_corr_EE = np.concatenate((m_corr_EE, val_m_corr_EE))
                    m_corr_BB = np.concatenate((m_corr_BB, val_m_corr_BB))

                index_corr += 1

        # take care of m-correction:
        m_corr = np.concatenate((m_corr_EE, m_corr_BB))
        # this is required for scaling of covariance matrix:
        m_corr_matrix = np.matrix(m_corr).T * np.matrix(m_corr)

        theory_BB = theory_BB.flatten() + theory_noise_BB.flatten()
        theory_EE = theory_EE.flatten() + theory_noise_EE.flatten()
        band_powers_theory = np.concatenate((theory_EE, theory_BB))

        #apply m-corrections also to covariance:
        # we want elementwise division!!!
        covariance = self.covariance / np.asarray(m_corr_matrix)

        # some numpy-magic for slicing:
        cov_sliced = covariance[np.ix_(self.indices_for_bands_to_use, self.indices_for_bands_to_use)]

        # invert covariance matrix:
        #inv_cov_sliced = np.linalg.inv(cov_sliced)

        # Eq. 16 from Heymans et al. 2013 (arxiv:1303.1808v1)
        # not necessary for analytical covariance!
        '''
        if self.use_debias_factor:
            params = len(self.indices_for_bands_to_use)
            debias_factor = (self.nmocks - params - 2.) / (self.nmocks - 1.)
        else:
            debias_factor = 1.

        inverse_covariance_debiased = debias_factor * inv_cov_sliced
        '''

        # m-correction is applied to DATA! Can also be marginalized over!
        difference_vector = (self.band_powers / m_corr) - band_powers_theory
        difference_vector = difference_vector[self.indices_for_bands_to_use]

        # Don't invert that matrix!
        #chi2 = difference_vector.T.dot(inv_cov_sliced.dot(difference_vector))
        # this is for running smoothly with MultiNest
        # (in initial checking of prior space, there might occur weird solutions)
        if np.isinf(band_powers_theory).any() or np.isnan(band_powers_theory).any():
            chi2 = 2e12
        else:
            # use a Cholesky decomposition instead:
            cholesky_transform = cholesky(cov_sliced, lower=True)
            yt = solve_triangular(cholesky_transform, difference_vector, lower=True)
            chi2 = yt.dot(yt)

        return -0.5 * chi2

    def baryon_feedback_bias_sqr(self, k, z, A_bary=1.):
        """

        Fitting formula for baryon feedback following equation 10 and Table 2 from J. Harnois-Deraps et al. 2014 (arXiv.1407.4301)

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

    def get_factor_IA(self, z, linear_growth_rate, amplitude, exponent):

        const = 5e-14 / self.small_h**2 # in Mpc^3 / M_sol

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

    # also for B-mode prediction:
    def get_theory(self, ells_sum, D_l, band_window_matrix, index_corr, band_type_is_EE=True):

        # these slice out the full EE --> EE and BB --> BB block of the full BWM!
        slicing_points_EE_x = (0, self.nzcorrs * self.band_offset_EE)
        slicing_points_EE_y = (0, self.nzcorrs * len(self.ells_intp))
        slicing_points_BB_x = (self.nzcorrs * self.band_offset_EE, self.nzcorrs * (self.band_offset_BB + self.band_offset_EE))
        slicing_points_BB_y = (self.nzcorrs * len(self.ells_intp), 2 * self.nzcorrs * len(self.ells_intp))

        if band_type_is_EE:
            slicing_points_x = slicing_points_EE_x
            slicing_points_y = slicing_points_EE_y
            band_offset = self.band_offset_EE
        else:
            slicing_points_x = slicing_points_BB_x
            slicing_points_y = slicing_points_BB_y
            band_offset = self.band_offset_BB

        #print(band_window_matrix)
        #print(band_window_matrix.shape)

        bwm_sliced = band_window_matrix[slicing_points_x[0]:slicing_points_x[1], slicing_points_y[0]:slicing_points_y[1]]

        #print(bwm)
        #print(bwm.shape)

        #ell_norm = ells_sum * (ells_sum + 1) / (2. * np.pi)

        bands = xrange(index_corr * band_offset, (index_corr + 1) * band_offset)

        D_avg = np.zeros(len(bands))

        for index_band, alpha in enumerate(bands):
            # jump along tomographic auto-correlations only:
            index_ell_low = int(index_corr * len(self.ells_intp))
            index_ell_high = int((index_corr + 1) * len(self.ells_intp))
            spline_w_alpha_l = itp.splrep(self.ells_intp, bwm_sliced[alpha, index_ell_low:index_ell_high])
            #norm_val = np.sum(itp.splev(ells_sum, spline_w_alpha_l))
            #print('Norm of W_al = {:.2e}'.format(norm_val))
            D_avg[index_band] = np.sum(itp.splev(ells_sum, spline_w_alpha_l) * D_l)

        return D_avg

    # model for self-induced B-modes due to resetting negative band powers
    # at each start of an iteration
    def get_B_mode_model(self, x, amp, exp):

        y = amp * x**exp

        return y

    def get_noise_prediction(self, ells_sum, A_noise, band_window_matrix, band_type_is_EE=True):

        # these slice out the full EE --> EE and BB --> BB block of the full BWM!
        slicing_points_EE_x = (0, self.nzcorrs * self.band_offset_EE)
        slicing_points_EE_y = (0, self.nzcorrs * len(self.ells_intp))
        slicing_points_BB_x = (self.nzcorrs * self.band_offset_EE, self.nzcorrs * (self.band_offset_BB + self.band_offset_EE))
        slicing_points_BB_y = (self.nzcorrs * len(self.ells_intp), 2 * self.nzcorrs * len(self.ells_intp))

        if band_type_is_EE:
            slicing_points_x = slicing_points_EE_x
            slicing_points_y = slicing_points_EE_y
            band_offset = self.band_offset_EE
        else:
            slicing_points_x = slicing_points_BB_x
            slicing_points_y = slicing_points_BB_y
            band_offset = self.band_offset_BB

        #print(band_window_matrix)
        #print(band_window_matrix.shape)

        bwm_sliced = band_window_matrix[slicing_points_x[0]:slicing_points_x[1], slicing_points_y[0]:slicing_points_y[1]]

        #print(bwm)
        #print(bwm.shape)

        ell_norm = ells_sum * (ells_sum + 1) / (2. * np.pi)

        pnoise = np.zeros(self.nzcorrs * band_offset)
        index_lin = 0
        for index_band in range(band_offset):
            for index_corr in range(self.nzcorrs):
                # jump along tomographic auto-correlations only:
                index_ell_low = int(index_corr * len(self.ells_intp))
                index_ell_high = int((index_corr + 1) * len(self.ells_intp))
                spline_w_alpha_l = itp.splrep(self.ells_intp, bwm_sliced[index_lin, index_ell_low:index_ell_high])
                pnoise[index_lin] = np.sum(itp.splev(ells_sum, spline_w_alpha_l) * A_noise[index_corr] * ell_norm)

                index_lin += 1

        return pnoise
