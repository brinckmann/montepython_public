##########################################################################################
# Euclid lensing, photometric galaxy clustering and cross-correlation (3x2pt) likelihood #
##########################################################################################

# - Based on an earlier euclid_lensing likelihood initiated by A. Audren and J. Lesgourgues 1210.7183
# - Improved by Sprenger et al. 1801.08331
# - Further developped to include clustering, cross-correlation and match IST:Fisher recipe by
#   S. Casas, M. Doerenkamp, J. Lesgourgues, L. Rathmann, Sabarish V., N. Schoeneberg
# - validated against CosmicFish and IST:Fisher in 2303.09451
# - further improved and generalised to massive neutrinos by S. Pamuk, S. Casas

from montepython.likelihood_class import Likelihood

from scipy.integrate import trapz
from scipy.interpolate import interp1d, RectBivariateSpline
from copy import copy

import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from time import time

from functools import partial

import numpy as np
import warnings
from scipy.special import erf

class euclid_photometric_alm(Likelihood):

    # Initialization performed a single time at the beginning of the run

    def __init__(self, path, data, command_line):
        self.debug_save  = False
        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # euclid_photometric_alm.zmax and for k up to euclid_photometric_alm.k_max_h_by_Mpc
        self.need_cosmo_arguments(data, {'output': 'mPk, dTk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.k_max_h_by_Mpc})

        # Define array of l values, evenly spaced in log scale,
        # with different lmax for WL, GC and XC
        if self.lmax_WL > self.lmax_GC:
            self.l_WL = np.logspace(np.log10(self.lmin), np.log10(self.lmax_WL), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_WL >= self.lmax_GC)[0])
            self.l_GC = self.l_WL[:self.idx_lmax+1]
            self.l_XC = self.l_WL[:self.idx_lmax+1]
            self.l_array = 'WL'

            self.ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            self.ell_jump = self.lmax_GC - self.lmin +1
            self.ells_GC = self.ells_WL[:self.ell_jump]
            self.ells_XC = self.ells_GC
        else:
            self.l_GC = np.logspace(np.log10(self.lmin), np.log10(self.lmax_GC), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_GC >= self.lmax_WL)[0])
            self.l_WL = self.l_GC[:self.idx_lmax+1]
            self.l_XC = self.l_GC[:self.idx_lmax+1]
            self.l_array = 'GC'

            self.ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            self.ell_jump = self.lmax_WL - self.lmin +1
            self.ells_WL = self.ells_GC[:self.ell_jump]
            self.ells_XC = self.ells_WL

        if self.debug_save :
            np.savetxt('ls.txt',self.l_GC)

        #################################################
        # Find galaxy distribution eta_i(z) in each bin #
        #################################################

        # Create the array that will contain the z boundaries for each bin.
        # Hard-coded, excepted the lowest and highest boundaries passed in euclid_photometric_alm.data
        self.z_bin_edge = np.array([self.zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, self.zmax])
        self.z_bin_center = np.array([(self.z_bin_edge[i]+self.z_bin_edge[i+1])/2 for i in range(self.nbin)])

        # Fill array of discrete z values
        self.z = np.linspace(self.zmin, self.zmax, num=self.nzmax)

        # Fill distribution for each bin (taking into account photo_z distribution)
        # eta_i = eta(z) * int dz_ph photoerror(z_ph|z) where the integral is over bin i (zi- < zp < zi+)

        # This will be eta_i(z)
        self.eta_z = np.zeros((self.nzmax, self.nbin), 'float64')

        # This will be: int dz_ph p_ph(z_ph|z) over a given bin range [zi-,zi+]. It is still a function of z.
        self.photoerror_z = np.zeros((self.nzmax, self.nbin), 'float64')

        for Bin in range(self.nbin):
            for nz in range(self.nzmax):
                z = self.z[nz]
                self.photoerror_z[nz,Bin] = self.photo_z_distribution(z,Bin+1)
                # eta_i(z) = eta(z) * [int dz_ph photoerror(z_ph|z)]
                self.eta_z[nz, Bin] = self.galaxy_distribution(z) * self.photoerror_z[nz,Bin]

        if self.debug_save : np.savetxt('./photoz.txt',self.photoerror_z)
        if self.debug_save : np.savetxt('./unnorm_nofz.txt',self.eta_z)

        # Normalize eta_i(z) to one
        for Bin in range(self.nbin):
            self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],self.z[:])

        if self.debug_save : np.savetxt('./n.txt',self.eta_z)

        # Number density of galaxies per bin in inverse square radian
        self.n_bar = self.gal_per_sqarcmn * (60.*180./np.pi)**2 / self.nbin

        ###########################
        # Add nuisance parameters #
        ###########################

        # For GC: bias parameters
        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            self.bias_names = []
            for ibin in range(self.nbin):
                self.bias_names.append('bias_'+str(ibin+1))
            self.nuisance += self.bias_names

        # For WL: intrinsic alignment parameters
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            self.nuisance += ['aIA', 'etaIA', 'betaIA']

            # Read the file for the IA- contribution
            lum_file = open(os.path.join(self.data_directory,'scaledmeanlum_E2Sa.dat'), 'r')
            content = lum_file.readlines()
            zlum = np.zeros((len(content)))
            lum = np.zeros((len(content)))
            for index in range(len(content)):
                line = content[index]
                zlum[index] = line.split()[0]
                lum[index] = line.split()[1]
            self.lum_func = interp1d(zlum, lum,kind='linear')

        # If requested: baryonic feedback parameters
        # (this is a minimal set, if needed, more parameters can be varied and passed to BCemu)
        if self.use_BCemu or (self.fit_different_data and self.data_use_BCemu):
            self.nuisance += ['log10Mc']
            self.nuisance += ['nu_Mc']

        #############
        # Read data #
        #############

        # If the fiducial file exists, read it. Otherwise it will be computed and written to a file in the function loglkl()
        self.fid_values_exist = False
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file+'.npz')
        if os.path.exists(fid_file_path):
            self.Cov_observ_dict = dict()
            self.fid_values_exist = True
            fid_file = np.load(fid_file_path)
            if fid_file['probe'] != self.probe:
                warnings.warn("Probes in fiducial file does not match the probes asked for.\n The fiducial Probe is {} and the probe asked for is {}.\n Please proceed with caution".format(fid_file['probe'],self.probe))
            try:
                if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                    l_WL = fid_file['ells_LL']
                    if not np.isclose(l_WL,self.l_WL).all():
                        raise Exception("Maximum multipole of WL has changed between fiducial and now.\n Fiducial lmax = {}, new lmax = {}. \n Please remove old fiducial and generate a new one".format(max(l_WL),max(self.l_WL)))
                    Cl_LL = fid_file['Cl_LL']
                    inter_LL = interp1d(l_WL,Cl_LL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_WL)
                    self.Cov_observ_dict["Cov_observ"] = inter_LL

                if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                    l_GC = fid_file['ells_GG']
                    if not np.isclose(l_GC,self.l_GC).all():
                        raise Exception("Maximum multipole of GC has changed between fiducial and now.\n Fiducial lmax = {}, new lmax = {}. \n Please remove old fiducial and generate a new one".format(max(l_GC),max(self.l_GC)))
                    Cl_GG = fid_file['Cl_GG']
                    inter_GG = interp1d(l_GC,Cl_GG,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_GC)
                    self.Cov_observ_dict["Cov_observ"] = inter_GG

                if 'WL_GCph_XC' in self.probe:
                    l_XC = fid_file['ells_GL']
                    Cl_GL = fid_file['Cl_GL']
                    inter_GL = interp1d(l_XC,Cl_GL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_XC)
                    inter_LG = np.transpose(inter_GL,(0,2,1))
                    if self.lmax_WL > self.lmax_GC:
                        self.Cov_observ_dict["Cov_observ"] = np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                        self.Cov_observ_dict["Cov_observ_high"] = inter_LL[self.ell_jump:,:,:]
                    else:
                        self.Cov_observ_dict["Cov_observ"] = np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                        self.Cov_observ_dict["Cov_observ_high"] = inter_GG[self.ell_jump:,:,:]

            except KeyError:
                raise KeyError("The probe asked for in the survey specifications is not in the fiducial file. \n Please remove old fiducial and generate a new one")
        else:
            if self.fit_different_data:
                self.use_BCemu = self.data_use_BCemu
                self.use_tracer = self.data_use_tracer

        return

    # Total galaxy distribution eta(z), unnormalized

    def galaxy_distribution(self, z):
        """
        Total galaxy distribution eta(z), unnormalized

        Modified by S. Clesse in March 2016 to add an optional form of n(z) motivated by ground based exp. (Van Waerbeke et al., 2013)
        See google doc document prepared by the Euclid IST - Splinter 2
        """
        zmean = 0.9
        z0 = zmean/np.sqrt(2)

        galaxy_dist = (z/z0)**2*np.exp(-(z/z0)**(1.5))

        return galaxy_dist

    # Photometric redhsift error p(z_ph|z) integrated over z_ph within one bin

    def photo_z_distribution(self, z, bin):
        """
        Photometric redhsift error p(z_ph|z) integrated over z_ph within one bin

        z:      physical galaxy redshift
        zph:    measured galaxy redshift
        """
        c0, z0, sigma_0 = 1.0, 0.1, 0.05
        cb, zb, sigma_b = 1.0, 0.0, 0.05
        f_out = 0.1

        if bin == 0 or bin >= 11:
            return None

        term1 = cb*f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))
        term2 =-cb*f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin    ]))/(sigma_0*(1+z)))
        term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))
        term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin    ]))/(sigma_b*(1+z)))

        return (term1+term2+term3+term4)/(2*c0*cb)

    # Compute Log(likelihood) = -chi2/2

    def loglkl(self, cosmo, data):

        if self.printtimes:
            t_start = time()

        # Relation between redshift z and comoving radius r,
        # inferred from cosmological module with the function z_of_r
        self.r = np.zeros(self.nzmax, 'float64')
        self.dzdr = np.zeros(self.nzmax, 'float64')
        self.r, self.dzdr = cosmo.z_of_r(self.z)

        # H(z)/c in 1/Mpc
        self.H_z = self.dzdr
        # H_0/c in 1/Mpc
        H0 = cosmo.h()/2997.92458

        # k_min and k_max in 1/Mpc
        self.kmin_in_inv_Mpc = self.k_min_h_by_Mpc * cosmo.h()
        self.kmax_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()

        # l = array of multipoles up to max(lmax_WL, lmax_GC)
        if self.l_array == 'WL':
            l = self.l_WL
        if self.l_array == 'GC':
            l = self.l_GC

        # k = corresponding array using Limber
        k =(l[:,None]+0.5)/self.r

        if self.printtimes:
            t_init = time()
            print("Initalisation Time:", t_init-t_start)

        ######################
        # Get power spectrum #
        ######################

        # Get non-linear and linear matter power spectrum P(k=(l+1/2)/r,z) in Mpc^3 from cosmological module
        Pk_m_nl_grid, k_grid, z_grid = cosmo.get_pk_and_k_and_z()
        Pk_m_l_grid, _, _ = cosmo.get_pk_and_k_and_z(nonlinear=False)

        # Order them by increasing redhsift / growing lookback time
        z_grid = z_grid[::-1]
        Pk_m_nl_grid = np.flip(Pk_m_nl_grid,axis=1)
        Pk_m_l_grid = np.flip(Pk_m_l_grid,axis=1)

        # Spline in view of interpolation
        Pk_m_nl_spline = RectBivariateSpline(k_grid,z_grid,Pk_m_nl_grid)
        Pk_m_l_spline = RectBivariateSpline(k_grid,z_grid,Pk_m_l_grid)

        # Spectra P(l,z) sampled at required k(l,z) values.
        # The spectra are zero when k(l,z) is outside of the range [kmin, kmax].
        Pk_m_nl = np.zeros_like(k, 'float64')
        Pk_m_l  = np.zeros_like(k, 'float64')
        for iz, zi in enumerate(self.z):
            pknn_mask = np.where((k[:,iz]>self.kmin_in_inv_Mpc) & (k[:,iz]<self.kmax_in_inv_Mpc))
            Pk_m_nl[pknn_mask, iz] = np.reshape(Pk_m_nl_spline(k[pknn_mask,iz],zi),Pk_m_nl[pknn_mask, iz].shape)
            Pk_m_l [pknn_mask, iz] = np.reshape(Pk_m_l_spline(k[pknn_mask,iz],zi),Pk_m_l [pknn_mask, iz].shape)

        # Non-linear spectrum P_NL(l,z) copied to Pk_WL and Pk_GC
        Pk_WL = copy(Pk_m_nl)
        Pk_GC = copy(Pk_m_nl)

        if self.printtimes:
            t_power = time()
            print("Power spectrum obtained in:", t_power-t_init)

        ########################
        # Boosts and emulators #
        ########################

        # Multiply only Pk_WL by baryonic feedback. You could multiply also Pk_GC depending on your physical assumptions.
        if self.use_BCemu:
            import baryonic_feedback
            # choice to only effect the Lensing power spectrum by baryonic effects
            Pk_WL *= baryonic_feedback.get_boost_baryonic_feedback(cosmo,data,self,k,self.z)

        # Pk_XC defined as geometric mean of Pk_WL and Pk_GC
        Pk_XC = np.sqrt(Pk_GC * Pk_WL)

        if self.printtimes:
            t_nonlinear = time()
            print("Nonlinear effects and Boosts obtained in", t_nonlinear - t_power)

        ############################
        # Get growth factor D(z,k) #
        ############################

        # Scale-independent case passed by cosmology module
        if self.scale_dependent_f == False:
            D_z= np.ones_like((self.nzmax), 'float64')
            for iz, zi in enumerate(self.z):
                D_z[iz] = cosmo.scale_independent_growth_factor(zi)
            D_z= D_z[None,:]

        # Scale-dependent case (e.g. with massive neutrinos) obtained directly from [P_NL(z)/P_NL(0)]^1/2
        elif self.scale_dependent_f ==True:
            D_z =np.ones_like(k, 'float64')
            for iz, zi in enumerate(self.z):
                pknn_mask = np.where((k[:,iz]>self.kmin_in_inv_Mpc) & (k[:,iz]<self.kmax_in_inv_Mpc))
                D_z[pknn_mask,iz] = np.reshape(np.sqrt(Pk_m_l_spline(k[pknn_mask,iz],zi) / Pk_m_l_spline(k[pknn_mask,iz],0)),D_z[pknn_mask,iz].shape)

        if self.printtimes:
            t_growth = time()
            print("Growthfactor obtained in", t_growth-t_nonlinear)

        ################################################
        # Window functions W_L(l,z,bin) and W_G(z,bin) #
        ################################################
        # in units of [W] = 1/Mpc

        # WL / cosmic shear case
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:

            # Contribution from matter power spectrum
            integral = 3./2.*H0**2. *cosmo.Omega_m()*self.r[None,:,None]*(1.+self.z[None,:,None])*self.eta_z.T[:,None,:]*(1-self.r[None,:,None]/self.r[None,None,:])
            W_gamma  = np.trapz(np.triu(integral),self.z,axis=-1).T

            # Contribution from intrinsic alignment
            W_IA = self.eta_z *self.H_z[:,None]
            C_IA = 0.0134
            A_IA = data.mcmc_parameters['aIA']['current']*(data.mcmc_parameters['aIA']['scale'])
            eta_IA = data.mcmc_parameters['etaIA']['current']*(data.mcmc_parameters['etaIA']['scale'])
            beta_IA = data.mcmc_parameters['betaIA']['current']*(data.mcmc_parameters['betaIA']['scale'])
            F_IA = (1.+self.z)**eta_IA * (self.lum_func(self.z))**beta_IA

            # Sum of the two
            W_L = W_gamma[None,:,:] - A_IA*C_IA*cosmo.Omega_m()*F_IA[None,:,None]/D_z[:,:,None] *W_IA[None,:,:]

        # GC / galaxy clustering case
        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:

            # Convert bias parameters passed as nuisance parameters into an array 'galaxy_bias' of biases at each z
            # See 2303.09451 for details on different prescriptions
            bias_values = np.zeros((self.nbin),'float64')
            for ibin in range(self.nbin):
                bias_values[ibin] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']
            # First prescription
            if self.bias_model == 'binned_constant' :
                galaxy_bias = bias_values[None,None,:]

            # Second prescription
            elif self.bias_model == 'binned' :
                biaspars = dict()
                for ibin in range(self.nbin):
                    biaspars['b'+str(ibin+1)] = bias_values[ibin]
                brang = range(1,len(self.z_bin_edge))
                last_bin_num = brang[-1]

                def binbis(zz):
                    lowi = np.where( self.z_bin_edge <= zz )[0][-1]
                    if zz >= self.z_bin_edge[-1] and lowi == last_bin_num:
                        bii = biaspars['b'+str(last_bin_num)]
                    else:
                        bii = biaspars['b'+str(lowi+1)]
                    return bii

                vbinbis = np.vectorize(binbis)
                galaxy_bias = vbinbis(self.z)[None,:,None]

            # Third prescription
            elif self.bias_model == 'interpld' :
                biasfunc = interp1d(self.z_bin_center, bias_values, bounds_error=False, fill_value="extrapolate")
                galaxy_bias = biasfunc(self.z)[None,:,None]

            # Handle the neutrino-induced scale-dependant bias following the prescription of 2405.06047
            if self.use_tracer == 'clustering':
                import neutrino_bias_model
                galaxy_bias = np.ones(self.lbin)[:,None,None] * galaxy_bias
                galaxy_bias *= np.sqrt(neutrino_bias_model.get_boost_neutrino_bias(cosmo, data, self, Pk_m_nl_grid, k, self.z)[:,:, None])

            # Now, compute the window functions
            W_G = np.zeros((self.nzmax, self.nbin), 'float64')
            W_G = galaxy_bias * self.H_z[None,:,None] * self.eta_z[None,:,:]

        if self.printtimes:
            t_window = time()
            print("window function obtained in:", t_window-t_nonlinear)

        ##################
        # Compute the Cl #
        ##################
        # dimensionless

        nell_WL = len(self.l_WL)
        nell_GC = len(self.l_GC)
        nell_XC = len(self.l_XC)

        # The indices are [ell, z, bin_i, bin_j] in the integrands (ending in _int)
        # and [ell, bin_i, bin_j] in the Cl

        # WL case
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_LL_int = W_L[:,:,:, None] * W_L[:,:, None,:] * Pk_WL[:,:, None, None] / self.H_z[None,:, None, None] / self.r[None,:, None, None] / self.r[None,:, None, None]
            Cl_LL     = trapz(Cl_LL_int,self.z,axis=1)[:nell_WL,:,:]

        # GC case
        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_GG_int = W_G[:,:,:, None] * W_G[:,:, None,:] * Pk_GC[:,:, None, None] / self.H_z[None,:, None, None] / self.r[None,:, None, None] / self.r[None,:, None, None]
            Cl_GG     = trapz(Cl_GG_int,self.z,axis=1)[:nell_GC,:,:]

        # cross-correlation XC case
        if 'WL_GCph_XC' in self.probe:
            Cl_LG_int = W_L[:,:,:, None] * W_G[:,:, None,:] * Pk_XC[:,:, None, None] / self.H_z[None,:, None, None] / self.r[None,:, None, None] / self.r[None,:, None, None]
            Cl_LG     = trapz(Cl_LG_int,self.z,axis=1)[:nell_XC,:,:]
            Cl_GL     = np.transpose(Cl_LG,(0,2,1))

        if self.printtimes:
            t_cell = time()
            print("Cell calculated in:", t_cell-t_window)

        #################################
        # In option, may plot Pk and Cl #
        #################################

        # do you want to save the power spectrum?
        if self.save_PS:
            debug_file_path = os.path.join( self.data_directory, 'euclid_photometric_alm_Pkz.npz')
            # loading the file yields P(k,z), k(ell,z), z
            np.savez(debug_file_path, Pk_WL=Pk_WL, Pk_GC=Pk_GC, k=k, z=self.z)
            print("Printed P(k,z)")

        # do you want to save the (noise-free) Cl?
        if self.save_Cell:
            debug_file_path = os.path.join(self.data_directory, 'euclid_photometric_alm_Cls.npz')
            if 'WL_GCph_XC' in self.probe:
                np.savez(debug_file_path, ells_LL=self.l_WL, ells_GG=self.l_GC, ells_GL=self.l_XC, Cl_LL = Cl_LL, Cl_GG = Cl_GG, Cl_GL = Cl_GL)
            if 'WL' in self.probe:
                np.savez(debug_file_path, ells_LL=self.l_WL, Cl_LL = Cl_LL)
            if 'GCph' in self.probe:
                np.savez(debug_file_path, ells_GG=self.l_GC, Cl_GG = Cl_GG)

        if self.printtimes:
            t_debug = time()
            print("Debug options obtained in:" , t_debug-t_cell)

        #####################
        # Add noise spectra #
        #####################
        # dimensionless

        # Constant noise spectra for each type
        self.noise = {
           'LL': self.rms_shear**2./self.n_bar,
           'LG': 0.,
           'GL': 0.,
           'GG': 1./self.n_bar}

        # Add noise to Cl
        for i in range(self.nbin):
            if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                Cl_LL[:,i,i] += self.noise['LL']
            if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                Cl_GG[:,i,i] += self.noise['GG']
            if 'WL_GCph_XC' in self.probe:
                Cl_GL[:,i,i] += self.noise['GL']
                Cl_LG[:,i,i] += self.noise['LG']

        ################################################
        # If needed, save computed Cl in fiducial file #
        ################################################

        if self.fid_values_exist is False:

            # Define fiducial file name
            fid_file_path = os.path.join(self.data_directory, self.fiducial_file)

            # Create a dictionary with all fiducial parameters
            fiducial_cosmo = dict()
            for key, value in data.mcmc_parameters.items():
                    fiducial_cosmo[key] = value['current']*value['scale']

            # Save this dictionary followed by all available Cl spectra
            if 'WL_GCph_XC' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_LL=self.l_WL, ells_GG=self.l_GC, ells_GL=self.l_XC, Cl_LL = Cl_LL, Cl_GG = Cl_GG, Cl_GL = Cl_GL)
            if 'WL' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_LL=self.l_WL, Cl_LL = Cl_LL)
            if 'GCph' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_GG=self.l_GC, Cl_GG = Cl_GG)

            # Write warning and abort. (The user should then start a new run that will use this fiducial file).
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        ##############################################################
        # Interpolate Cl at each l to build theory covariance matrix #
        ##############################################################

        # for each type WL, GC and XC, we do a spline interpolation of Cl spectra at each integer multipole l
        # The Cov_theory_dict contains both the full a_lm covariance constructed like a block matrix,
        # and the reduced matrices that appear when doing scale cuts.
        Cov_theory_dict = dict()

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:

            # If there is WL in the problem, we compute C_l^LL at each l
            inter_LL = interp1d(self.l_WL,Cl_LL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_WL)

            # We build tentatively a covariance matrix as if there would be only WL in the problem
            # (otherwise, the array will be overwritten later on).
            # The array Cov_theory_dict["Cov_theory"] of [l, bin1, bin2] contains C_l^[bin1, bin2] up to l_max_WL
            Cov_theory_dict["Cov_theory"] = inter_LL
            ells = self.ells_WL

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            inter_GG = interp1d(self.l_GC,Cl_GG,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_GC)
            Cov_theory_dict["Cov_theory"] = inter_GG
            ells = self.ells_GC

        if 'WL_GCph_XC' in self.probe:
            inter_GL = interp1d(self.l_XC,Cl_GL,axis=0, kind='cubic',fill_value="extrapolate")(self.ells_XC)
            inter_LG = np.transpose(inter_GL,(0,2,1))
            if self.lmax_WL > self.lmax_GC:
                Cov_theory_dict["Cov_theory"] = np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                Cov_theory_dict["Cov_theory_high"] = inter_LL[self.ell_jump:,:,:]
                ells = self.ells_WL
            else:
                Cov_theory_dict["Cov_theory"] = np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                Cov_theory_dict["Cov_theory_high"] = inter_GG[self.ell_jump:,:,:]
                ells = self.ells_GC

        if self.printtimes:
            t_spline = time()
            print("Covariance obtained in:", t_spline-t_debug)

        #############################################################
        # Compute likelihood, optionally adding a theoretical error #
        #############################################################

        # The treatment of theoretical errors is done as described in
        # [1210.2194], Appendix B4 (see also appendix A.4).

        # The idea is to marginalise over a global uncorrelated error
        # stemming from an uncertainty in the predicted power spectrum.

        # In this method we add a new nuisance parameter epsilon_l for
        # each multipole. This parameter obeys to a normal
        # distribution with unit variance, and mutiplies a
        # 'theoretical error estimate' that gets added to C_ij(l) (the
        # predicted C_ij(l), not the observed one). Then, epsilon_l is
        # minimized (as a proxy for marginalisation) at the level of
        # the likelihood.

        # The 'error standard deviation' is represented by a matrix
        # R_ij(l) corresponding to a shift of the matter power
        # spectrum P(k,z) by a relative theoretical error [Delta
        # P(k,z)]/P(k,z). This matrix can be computed for a given
        # alpha(k,z) in the function
        # theoretical_errors.get_covariance_error().

        # The matrix R_ij(l) is stored in the dictionary T_Rerr_dict
        # obeying the same format as Cov_theory_dict. T_Rerr_dict = 0
        # amounts to 'no theoretical error'.
        T_Rerr_dict = dict()
        T_Rerr_dict["T_Rerr"] = np.zeros_like(Cov_theory_dict["Cov_theory"])
        if 'WL_GCph_XC' in self.probe:
            T_Rerr_dict["T_Rerr_high"] = np.zeros_like(Cov_theory_dict["Cov_theory_high"])

        # To prepare the computation of the chi2 given the theory and
        # observed (=fiducial) covariance matrices, performed by the
        # function compute_chisq(), we define a function
        # pcompute_chisq() in which several arguments are already
        # passed. The new function pcompute_chisq only needs to be
        # passed the (yet unkown) epsilon parameters.
        pcompute_chisq =  partial(self.compute_chisq, ells = ells, Cov_observ_dict = self.Cov_observ_dict, Cov_theory_dict = Cov_theory_dict, T_Rerr_dict = T_Rerr_dict)

        # If needed, minimize over each nuisance parameter epsilon_l
        # (abbreviated as eps_l) accounting for the theoretical
        # error. Fixing eps_l=0 amounts to `no theoretical error'.
        eps_l = np.zeros_like(ells)
        if self.theoretical_error != False:
            import theoretical_errors
            # See the theoretical_errors file for a more detailed explanation of each step

            # Compute the 'error standard deviation' R_ij(l), for a
            # given assumption on the theoretical error [Delta P(k,z)]
            # on the power spectrum P(k,z), and for a few multipole l
            El_dict = theoretical_errors.get_covariance_error(cosmo, data, self, k, Pk_WL, Pk_GC, Pk_XC, W_L, W_G)

            # Use spline interpolation to get R_ij(l) at every l.
            # The update of T_Rerr_dict updates pcompute_chisq as well
            T_Rerr_dict.update(theoretical_errors.spline_error(self, El_dict))

            # Minimize the chi2 over each eps_l and return the vector
            # of optimal eps_l values
            eps_l = theoretical_errors.minimize_chisq(self, self.compute_chisq, ells, self.Cov_observ_dict, Cov_theory_dict, T_Rerr_dict)

        # Computes the chi2. If no theoretical errors are assumed,
        # eps_l = 0 and the chi2 is computed in the normal way.
        chi2 = pcompute_chisq(eps_l)

        # Timing for debugging and optimisation
        if self.printtimes:
            t_lkl = time()
            print("Likelihood calculated in:" ,t_lkl-t_spline)
            print("Total time taken:", t_lkl-t_start)

        # finally return log(lkl) = -chi2/2
        return -chi2/2.

    ##############################

    # Compute the chi2 = - 2 log(lkl) for given theory/observation covariance matrices
    # (and optionally nuisance parameters accounting for theoretical error)

    def compute_chisq(self, eps_l, ells, Cov_observ_dict, Cov_theory_dict, T_Rerr_dict):

        # The observation covariance matrix C_ij(l) accounts for the
        # data, i.e. the fiducial model.
        Cov_observ = Cov_observ_dict["Cov_observ"]

        # The theory covariance matrix C_ij(l) accounts for each
        # assumed model.
        Cov_theory = Cov_theory_dict["Cov_theory"]

        # The theoretical error standard deviation R_ij(l) is defined
        # as the shift in C_ij(l) produced by a shift of the power
        # spectrum P(k,z) by some assumed relative error function
        # [Delta P(k,z)]/P(k,z). It is also computed for each assumed
        # model.
        T_Rerr = T_Rerr_dict["T_Rerr"]

        # Infer the number of z bins and l_max from the dimension of
        # C_ij(l).
        nbin = Cov_observ.shape[1]
        ell_jump = Cov_observ.shape[0]

        # If theoretical errors are requested, add the theoretical
        # error to the theory covariance matrix. The theoretical error
        # is given by the 'error standard deviation' R_ij(l) times the
        # minimized 'error amplitude' eps_l. If no theoretical error
        # was requested, eps_l = 0 and the following lien is still
        # valid.
        shifted_Cov = Cov_theory + eps_l[:ell_jump, None, None] * T_Rerr

        # Compute the determinant of the theory covariance matrix.
        dtilde_the = np.linalg.det(shifted_Cov)

        # Compute the determinant of the observed covariance matrix.
        d_obs = np.linalg.det(Cov_observ)

        # Compute the mixed determinant (of theory covariance matrix
        # with one column replaced by the observed one, with a sum
        # over the nbin possibilities).
        dtilde_mix = np.zeros_like(dtilde_the)
        for i in range(nbin):
            newCov = np.copy(shifted_Cov)
            newCov[:, i] = Cov_observ[:, :, i]
            dtilde_mix += np.linalg.det(newCov)

        # This unit matrix multiplied by the covariance matrix
        # dimension will be used to normalise the likelihood to
        # L/L_max, such that Delta chi2=0 at the fiducial
        N = np.ones_like(ells) * nbin

        # If the probe includes 3x2pt, calculate the contribution of
        # multipoles l > min(l_max_LL, l_max_GG) for which we should
        # only consider the LL or GG auto-correlation.
        if "WL_GCph_XC" in self.probe:

            # the computation of the chi2 from these multipoles is
            # similar, just using matrices with smaller
            # dimension.
            Cov_observ_high = Cov_observ_dict["Cov_observ_high"]
            Cov_theory_high = Cov_theory_dict["Cov_theory_high"]
            T_Rerr_high = T_Rerr_dict["T_Rerr_high"]

            nbin = Cov_observ_high.shape[1]

            # Add theoretical error
            shifted_Cov_high = Cov_theory_high + eps_l[ell_jump:, None, None] * T_Rerr_high

            # Compute the determinants
            dtilde_the_high = np.linalg.det(shifted_Cov_high)
            d_obs_high = np.linalg.det(Cov_observ_high)
            dtilde_mix_high = np.zeros_like(dtilde_the_high)
            for i in range(nbin):
                newCov = np.copy(shifted_Cov_high)
                newCov[:, i] = Cov_observ_high[:, :, i]
                dtilde_mix_high += np.linalg.det(newCov)

            # Append the vector of determinants acounting for l >
            # min(l_max_LL, l_max_GG) to the list of determinants
            # accounting for l < min(l_max_LL, l_max_GG)
            N[ell_jump:] = nbin
            dtilde_the = np.concatenate([dtilde_the, dtilde_the_high])
            d_obs = np.concatenate([d_obs, d_obs_high])
            dtilde_mix = np.concatenate([dtilde_mix, dtilde_mix_high])

        # Compute and return the chi2. If eps_l is non-zero, the last
        # term eps_l**2 accounts for the fact that we assume a normal
        # distribution with unit variance for eps_l.
        return np.sum((2 * ells + 1) * self.fsky * ((dtilde_mix / dtilde_the) + np.log(dtilde_the / d_obs) - N) + np.power(eps_l, 2))
