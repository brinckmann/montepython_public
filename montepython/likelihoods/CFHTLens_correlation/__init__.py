##############################################################
# likelihood for the CFHTLens correlation function           #
##############################################################
#
# Set up by Antonio J. Cuesta and J. Lesgourgues, by adapting
# Benjamin Audren's Monte Python likelihood euclid_lensing and
# Adam J Moss's CosmoMC likelihood for weak lensing
# (adapted itself from JL's CosmoMC likelihood for the COSMOS)
#
# Designed for using data from Heymans et al. 1303.1808
#
##############################################################

from montepython.likelihood_class import Likelihood
import io_mp

import scipy.integrate
from scipy import interpolate as itp
from scipy import special
import os
import numpy as np
import math

try:
    xrange
except NameError:
    xrange = range

class CFHTLens_correlation(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z) and for k up to k_max
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': self.k_max_h_by_Mpc})

        # Compute non-linear power spectrum if requested
        if (self.use_halofit):
            self.need_cosmo_arguments(data, {'non linear':'halofit'})

        # Define array of l values, and initialize them
        # It is a logspace
        # find nlmax in order to reach lmax with logarithmic steps dlnl
        self.nlmax = np.int(np.log(self.lmax)/self.dlnl)+1
        # redefine slightly dlnl so that the last point is always exactly lmax
        self.dlnl = np.log(self.lmax)/(self.nlmax-1)
        self.l = np.exp(self.dlnl*np.arange(self.nlmax))

        # Read dn_dz from window files
        self.z_p = np.zeros(self.nzmax)
        zptemp = np.zeros(self.nzmax)
        self.p = np.zeros((self.nzmax, self.nbin))
        for i in xrange(self.nbin):
            window_file_path = os.path.join(
                self.data_directory, self.window_file[i])
            if os.path.exists(window_file_path):
                zptemp = np.loadtxt(window_file_path, usecols=[0])
                if (i > 0 and np.sum((zptemp-self.z_p)**2) > 1e-6):
                    raise io_mp.LikelihoodError(
                        "The redshift values for the window files "
                        "at different bins do not match")
                self.z_p = zptemp
                self.p[:, i] = np.loadtxt(window_file_path, usecols=[1])
            else:
                raise io_mp.LikelihoodError("File not found:\n %s"%window_file_path)

        # Read measurements of xi+ and xi-
        nt = (self.nbin)*(self.nbin+1)//2
        self.theta_bins = np.zeros(2*self.ntheta)
        self.xi_obs = np.zeros(self.ntheta*nt*2)
        xipm_file_path = os.path.join(
            self.data_directory, self.xipm_file)
        if os.path.exists(xipm_file_path):
            self.theta_bins = np.loadtxt(xipm_file_path)[:, 0]
            if (np.sum(
                (self.theta_bins[:self.ntheta] -
                    self.theta_bins[self.ntheta:])**2) > 1e-6):
                raise io_mp.LikelihoodError(
                    "The angular values at which xi+ and xi- "
                    "are observed do not match")
            temp = np.loadtxt(xipm_file_path)[:, 1:]
        else:
            raise io_mp.LikelihoodError("File not found:\n %s"%xipm_file_path)

        k = 0
        for j in xrange(nt):
            for i in xrange(2*self.ntheta):
                self.xi_obs[k] = temp[i, j]
                k = k + 1

        # Read covariance matrix
        ndim = (self.ntheta)*(self.nbin)*(self.nbin+1)
        covmat = np.zeros((ndim, ndim))
        covmat_file_path = os.path.join(self.data_directory, self.covmat_file)
        if os.path.exists(covmat_file_path):
            covmat = np.loadtxt(covmat_file_path)
        else:
            raise io_mp.LikelihoodError("File not found:\n %s"%covmat_file_path)

        covmat = covmat/self.ah_factor

        # Read angular cut values (OPTIONAL)
        if (self.use_cut_theta):
            cut_values = np.zeros((self.nbin, 2))
            cutvalues_file_path = os.path.join(
                self.data_directory, self.cutvalues_file)
            if os.path.exists(cutvalues_file_path):
                cut_values = np.loadtxt(cutvalues_file_path)
            else:
                raise io_mp.LikelihoodError("File not found:\n %s"%cutvalues_file_path)

        # Normalize selection functions
        self.p_norm = np.zeros(self.nbin, 'float64')
        for Bin in xrange(self.nbin):
            self.p_norm[Bin] = np.sum(0.5*(
                self.p[1:, Bin]+self.p[:-1, Bin])*(
                self.z_p[1:]-self.z_p[:-1]))

        # Compute theta mask
        if (self.use_cut_theta):
            mask = np.zeros(2*nt*self.ntheta)
            iz = 0
            for izl in xrange(self.nbin):
                for izh in xrange(izl, self.nbin):
                    # this counts the bin combinations
                    # iz=1 =>(1,1), iz=2 =>(1,2) etc
                    iz = iz + 1
                    for i in xrange(self.ntheta):
                        j = (iz-1)*2*self.ntheta
                        xi_plus_cut = max(
                            cut_values[izl, 0], cut_values[izh, 0])
                        xi_minus_cut = max(
                            cut_values[izl, 1], cut_values[izh, 1])
                        if (self.theta_bins[i] > xi_plus_cut):
                            mask[j+i] = 1
                        if (self.theta_bins[i] > xi_minus_cut):
                            mask[self.ntheta + j+i] = 1
        else:
            mask = np.ones(2*nt*self.ntheta)

        self.num_mask = int(np.sum(mask))
        self.mask_indices = np.zeros(self.num_mask)
        j = 0
        for i in xrange(self.ntheta*nt*2):
            if (mask[i] == 1):
                self.mask_indices[j] = i
                j = j+1
        self.mask_indices = np.int32(self.mask_indices)
        # Precompute masked inverse
        self.wl_invcov = np.zeros((self.num_mask, self.num_mask))
        self.wl_invcov = covmat[self.mask_indices][:, self.mask_indices]
        self.wl_invcov = np.linalg.inv(self.wl_invcov)

        # Fill array of discrete z values
        # self.z = np.linspace(0, self.zmax, num=self.nzmax)

        ################
        # Noise spectrum
        ################

        # Number of galaxies per steradian
        self.noise = 3600.*self.gal_per_sqarcmn*(180./math.pi)**2

        # Number of galaxies per steradian per bin
        self.noise = self.noise/self.nbin

        # Noise spectrum (diagonal in bin*bin space, independent of l and Bin)
        self.noise = self.rms_shear**2/self.noise

        ################################################
        # discrete theta values (to convert C_l to xi's)
        ################################################

        thetamin = np.min(self.theta_bins)*0.8
        thetamax = np.max(self.theta_bins)*1.2

        self.nthetatot = np.ceil(math.log(thetamax/thetamin)/self.dlntheta) + 1
        self.nthetatot = np.int32(self.nthetatot)
        self.theta = np.zeros(self.nthetatot, 'float64')
        self.a2r = math.pi/(180.*60.)

        # define an array of theta's
        for it in xrange(self.nthetatot):
            self.theta[it] = thetamin*math.exp(self.dlntheta*it)

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
        self.g = np.zeros((self.nzmax, self.nbin), 'float64')
        self.pk = np.zeros((self.nlmax, self.nzmax), 'float64')
        self.k_sigma = np.zeros(self.nzmax, 'float64')
        self.alpha = np.zeros((self.nlmax, self.nzmax), 'float64')
        if 'epsilon' in self.use_nuisance:
            self.E_th_nu = np.zeros((self.nlmax, self.nzmax), 'float64')
        self.nbin_pairs = self.nbin*(self.nbin+1)//2
        self.Cl_integrand = np.zeros((self.nzmax, self.nbin_pairs), 'float64')
        self.Cl = np.zeros((self.nlmax, self.nbin_pairs), 'float64')
        if self.theoretical_error != 0:
            self.El_integrand = np.zeros((self.nzmax, self.nbin_pairs),'float64')
            self.El = np.zeros((self.nlmax, self.nbin_pairs), 'float64')
        self.spline_Cl = np.empty(self.nbin_pairs, dtype=(list, 3))
        self.xi1 = np.zeros((self.nthetatot, self.nbin_pairs), 'float64')
        self.xi2 = np.zeros((self.nthetatot, self.nbin_pairs), 'float64')
        self.Cll = np.zeros((self.nbin_pairs,self.nl), 'float64')
        self.BBessel0 = np.zeros(self.nl, 'float64')
        self.BBessel4 = np.zeros(self.nl, 'float64')
        self.xi1_theta = np.empty(self.nbin_pairs, dtype=(list, 3))
        self.xi2_theta = np.empty(self.nbin_pairs, dtype=(list, 3))
        self.xi = np.zeros(np.size(self.xi_obs), 'float64')

        return

    def loglkl(self, cosmo, data):

        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        self.r, self.dzdr = cosmo.z_of_r(self.z_p)

        # Compute now the selection function p(r) = p(z) dz/dr normalized
        # to one. The np.newaxis helps to broadcast the one-dimensional array
        # dzdr to the proper shape. Note that p_norm is also broadcasted as
        # an array of the same shape as p_z
        self.p_r = self.p*(self.dzdr[:, np.newaxis]/self.p_norm)

        # Compute function g_i(r), that depends on r and the bin
        # g_i(r) = 2r(1+z(r)) int_r^+\infty drs p_r(rs) (rs-r)/rs
        for Bin in xrange(self.nbin):
            for nr in xrange(1, self.nzmax-1):
                fun = self.p_r[nr:, Bin]*(self.r[nr:]-self.r[nr])/self.r[nr:]
                self.g[nr, Bin] = np.sum(0.5*(
                    fun[1:]+fun[:-1])*(self.r[nr+1:]-self.r[nr:-1]))
                self.g[nr, Bin] *= 2.*self.r[nr]*(1.+self.z_p[nr])

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        kmax_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()
        for index_l in xrange(self.nlmax):
            for index_z in xrange(1, self.nzmax):
#                if (self.l[index_l]/self.r[index_z] > self.k_max):
#                    raise io_mp.LikelihoodError(
#                        "you should increase CFHTLens_correlation.k_max up"
#                        " to at least %g" % (self.l[index_l]/self.r[index_z]))
#                self.pk[index_l, index_z] = cosmo.pk(
#                    self.l[index_l]/self.r[index_z], self.z_p[index_z])
                k_in_inv_Mpc =  self.l[index_l]/self.r[index_z]
                if (k_in_inv_Mpc > kmax_in_inv_Mpc):
                    self.pk[index_l, index_z] = 0.0
                else:
                    self.pk[index_l, index_z] = cosmo.pk(k_in_inv_Mpc, self.z_p[index_z])

        # Recover the non_linear scale computed by halofit. If no scale was
        # affected, set the scale to one, and make sure that the nuisance
        # parameter epsilon is set to zero
        if (cosmo.nonlinear_method == 0):
            self.k_sigma[:] = 1.e6
        else:
            self.k_sigma = cosmo.nonlinear_scale(self.z_p, self.nzmax)

        # Define the alpha function, that will characterize the theoretical
        # uncertainty. Chosen to be 0.001 at low k, raise between 0.1 and 0.2
        # to self.theoretical_error
        if self.theoretical_error != 0:
            for index_l in xrange(self.nlmax):
                k = self.l[index_l]/self.r[1:]
                self.alpha[index_l, 1:] = np.log(1.+k[:]/self.k_sigma[1:])/(
                    1.+np.log(1.+k[:]/self.k_sigma[1:]))*self.theoretical_error

        # recover the e_th_nu part of the error function
        e_th_nu = self.coefficient_f_nu*cosmo.Omega_nu/cosmo.Omega_m()

        # Compute the Error E_th_nu function
        if 'epsilon' in self.use_nuisance:
            for index_l in xrange(1, self.nlmax):
                self.E_th_nu[index_l, :] = np.log(
                    1.+self.l[index_l]/self.k_sigma[:]*self.r[:]) / (
                    1.+np.log(1.+self.l[index_l]/self.k_sigma[:]*self.r[:]))*e_th_nu

        # Add the error function, with the nuisance parameter, to P_nl_th, if
        # the nuisance parameter exists
                for index_l in xrange(self.nlmax):
                    epsilon = data.mcmc_parameters['epsilon']['current']*(
                        data.mcmc_parameters['epsilon']['scale'])
                    self.pk[index_l, :] *= (1.+epsilon*self.E_th_nu[index_l, :])

        # Start loop over l for computation of C_l^shear
        # Start loop over l for computation of E_l
        for il in xrange(self.nlmax):
            # find Cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
            for Bin1 in xrange(self.nbin):
                for Bin2 in xrange(Bin1,self.nbin):
                    self.Cl_integrand[1:, self.one_dim_index(Bin1,Bin2)] = self.g[1:, Bin1]*self.g[1:, Bin2]/(self.r[1:]**2)*self.pk[il, 1:]
                    if self.theoretical_error != 0:
                        self.El_integrand[1:, self.one_dim_index(Bin1, Bin2)] = self.g[1:, Bin1]*(self.g[1:, Bin2])/(self.r[1:]**2)*self.pk[il, 1:]*self.alpha[il, 1:]

            # Integrate over r to get C_l^shear_ij = P_ij(l)
            # C_l^shear_ij = 9/16 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r)
            # g_j(r) /r**2) P(k=l/r,z(r)) dr
            # It is then multiplied by 9/16*Omega_m**2
            # and then by (h/2997.9)**4 to be dimensionless
            # (since P(k)*dr is in units of Mpc**4)
            for Bin in xrange(self.nbin_pairs):
                self.Cl[il, Bin] = np.sum(0.5*(self.Cl_integrand[1:, Bin] + self.Cl_integrand[:-1, Bin])*(self.r[1:]-self.r[:-1]))
                self.Cl[il, Bin] *= 9./16.*(cosmo.Omega_m())**2
                self.Cl[il, Bin] *= (cosmo.h()/2997.9)**4

                if self.theoretical_error != 0:
                    self.El[il, Bin] = np.sum(0.5*(self.El_integrand[1:, Bin] + self.El_integrand[:-1, Bin])*(self.r[1:]-self.r[:-1]))
                    self.El[il, Bin] *= 9./16.*(cosmo.Omega_m())**2
                    self.El[il, Bin] *= (cosmo.h()/2997.9)**4

            for Bin1 in xrange(self.nbin):
                self.Cl[il, self.one_dim_index(Bin1, Bin1)] += self.noise

        # Spline Cl[il,Bin1,Bin2] along l
        for Bin in xrange(self.nbin_pairs):
            self.spline_Cl[Bin] = list(itp.splrep(self.l, self.Cl[:, Bin]))

        # Interpolate Cl at values lll and store results in Cll
        for Bin in xrange(self.nbin_pairs):
            self.Cll[Bin,:] = itp.splev(self.lll[:], self.spline_Cl[Bin])

        # Start loop over theta values
        for it in xrange(self.nthetatot):
            ilmax = self.il_max[it]

            self.BBessel0[:ilmax] = special.j0(self.lll[:ilmax]*self.theta[it]*self.a2r)
            self.BBessel4[:ilmax] = special.jv(4,self.lll[:ilmax]*self.theta[it]*self.a2r)

            # Here is the actual trapezoidal integral giving the xi's:
            # - in more explicit style:
            # for Bin in xrange(self.nbin_pairs):
            #     for il in xrange(ilmax):
            #         self.xi1[it, Bin] = np.sum(self.ldl[il]*self.Cll[Bin,il]*self.BBessel0[il])
            #         self.xi2[it, Bin] = np.sum(self.ldl[il]*self.Cll[Bin,il]*self.BBessel4[il])
            # - in more compact and vectorizable style:
            self.xi1[it, :] = np.sum(self.ldl[:ilmax]*self.Cll[:,:ilmax]*self.BBessel0[:ilmax],axis=1)
            self.xi2[it, :] = np.sum(self.ldl[:ilmax]*self.Cll[:,:ilmax]*self.BBessel4[:ilmax],axis=1)

        # normalisation of xi's
        self.xi1 = self.xi1/(2.*math.pi)
        self.xi2 = self.xi2/(2.*math.pi)

        # Spline the xi's
        for Bin in xrange(self.nbin_pairs):
            self.xi1_theta[Bin] = list(itp.splrep(self.theta, self.xi1[:,Bin]))
            self.xi2_theta[Bin] = list(itp.splrep(self.theta, self.xi2[:,Bin]))

        # Get xi's in same column vector format as the data
        #iz = 0
        #for Bin in xrange(self.nbin_pairs):
        #    iz = iz + 1  # this counts the bin combinations
        #    for i in xrange(self.ntheta):
        #        j = (iz-1)*2*self.ntheta
        #        self.xi[j+i] = itp.splev(
        #            self.theta_bins[i], self.xi1_theta[Bin])
        #        self.xi[self.ntheta + j+i] = itp.splev(
        #            self.theta_bins[i], self.xi2_theta[Bin])
        # or in more compact/vectorizable form:
        iz = 0
        for Bin in xrange(self.nbin_pairs):
            iz = iz + 1  # this counts the bin combinations
            j = (iz-1)*2*self.ntheta
            self.xi[j:j+self.ntheta] = itp.splev(self.theta_bins[:self.ntheta], self.xi1_theta[Bin])
            self.xi[j+self.ntheta:j+2*self.ntheta] = itp.splev(self.theta_bins[:self.ntheta], self.xi2_theta[Bin])

        # final chi2
        vec = self.xi[self.mask_indices] - self.xi_obs[self.mask_indices]
        chi2 = np.dot(vec, np.dot(self.wl_invcov, vec))

        return -chi2/2.

    #######################################################################################################
    # This function is used to convert 2D sums over the two indices (Bin1, Bin2) of an N*N symmetric matrix
    # into 1D sums over one index with N(N+1)/2 possible values
    def one_dim_index(self,Bin1,Bin2):
        if Bin1 <= Bin2:
            return Bin2+self.nbin*Bin1-(Bin1*(Bin1+1))//2
        else:
            return Bin1+self.nbin*Bin2-(Bin2*(Bin2+1))//2
