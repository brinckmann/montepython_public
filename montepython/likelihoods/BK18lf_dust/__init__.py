"""
.. module:: BK18
    :synopsis: BK18 likelihood

.. moduleauthor:: Thomas Tram <thomas.tram@phys.au.dk>
Last updated 4 April, 2024. Based on the CosmoMC module.
"""
import numpy as np
import pandas as pd
import scipy.linalg as la
import montepython.io_mp as io_mp
import os
import functools
from collections import defaultdict
from montepython.likelihood_class import Likelihood_sn

T_CMB = 2.72548     #CMB temperature
h = 6.62606957e-34     #Planck's constant
kB = 1.3806488e-23     #Boltzmann constant
Ghz_Kelvin = h/kB*1e9  #GHz Kelvin conversion

class BK18lf_dust(Likelihood_sn):

    def __init__(self, path, data, command_line):
        # Unusual construction, since the data files are not distributed
        # alongside BK18 (size problems)
        try:
            # Read the .dataset file specifying the data.
            super(BK18lf_dust, self).__init__(path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The BK18 data files were not found. Please download the "
                "following link "
                "http://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz"
                ", extract it, and copy the `" + type(self).__name__ + "` folder inside "
                "`BK18_cosmomc/data/` to `your_montepython/data/`")

        # Require tensor modes from CLASS as well as nonlinear lensing.
        # Nonlinearities enhance the B-mode power spectrum by more than 6%
        # at l>100. (Even more at l>2000, but not relevant to BICEP.)
        # See http://arxiv.org/abs/astro-ph/0601594.
        arguments = {
            'output': 'tCl pCl lCl',
            'lensing': 'yes',
            'modes': 's, t',
            'l_max_scalars': 2000,
            'k_max_tau0_over_l_max': 7.0,
            'non linear':'HALOFIT' if self.do_nonlinear else '',
            'accurate_lensing':1,
            'l_max_tensors': self.cl_lmax}
        self.need_cosmo_arguments(data, arguments)

        map_names_used = self.map_names_used.split()
        map_fields = self.map_fields.split()
        map_names = self.map_names.split()
        self.map_fields_used = [maptype for i, maptype in enumerate(map_fields) if map_names[i] in map_names_used]

        nmaps = len(map_names_used)
        ncrossmaps = nmaps*(nmaps+1)//2
        nbins = int(self.nbins)

        ## This constructs a different flattening of triangular matrices.
        ## v = [m for n in range(nmaps) for m in range(n,nmaps)]
        ## w = [m for n in range(nmaps) for m in range(nmaps-n)]
        ## # Store the indices in a tuple of integer arrays for later use.
        ## self.flat_to_diag = (np.array(v),np.array(w))

        # We choose the tril_indices layout for flat indexing of the triangular matrix
        self.flat_to_diag = np.tril_indices(nmaps)
        self.diag_to_flat = np.zeros((nmaps,nmaps),dtype='int')
        # It is now easy to generate an array with the corresponding flattened indices. (We only fill the lower triangular part.)
        self.diag_to_flat[self.flat_to_diag] = list(range(ncrossmaps))

        # Read in bandpasses
        self.ReadBandpasses()

        # Read window bins; for BK18, the window extends to lmax=999, although l_max is set to 600 in the .dataset file
        self.window_data = np.zeros((int(self.nbins),int(self.cl_lmax),ncrossmaps))
        # Retrieve mask and index permutation of windows:
        indices, mask = self.GetIndicesAndMask(self.bin_window_in_order.split())
        for k in range(nbins):
            windowfile = os.path.join(self.data_directory, self.bin_window_files.replace('%u',str(k+1)))
            tmp = pd.read_table(windowfile,comment='#',sep=' ',header=None, index_col=0).to_numpy()
            print(k, tmp.shape)
            # Apply mask
            tmp = tmp[:,mask]
            print(tmp.shape)
            
            # Permute columns and store this bin
            self.window_data[k][:,indices] = tmp[:int(self.cl_lmax)]
        # print('window_data',self.window_data.shape)

        #Read covmat fiducial
        # Retrieve mask and index permutation for a single bin.
        indices, mask = self.GetIndicesAndMask(self.covmat_cl.split())
        # Extend mask and indices. Mask just need to be copied, indices needs to be increased:
        superindices = []
        supermask = []
        for k in range(nbins):
            superindices += [idx+k*ncrossmaps for idx in indices]
            supermask += list(mask)
        supermask = np.array(supermask)

        tmp = pd.read_table(os.path.join(self.data_directory, self.covmat_fiducial),comment='#',sep=' ',header=None,skipinitialspace=True).to_numpy()
        # Apply mask:
        tmp = tmp[:,supermask][supermask,:]
        print('Covmat read with shape',tmp.shape)
        # Store covmat in correct order
        self.covmat = np.zeros((nbins*ncrossmaps,nbins*ncrossmaps))
        for index_tmp, index_covmat in enumerate(superindices):
            self.covmat[index_covmat,superindices] = tmp[index_tmp,:]

        #Compute inverse and store
        self.covmat_inverse = la.inv(self.covmat)
        # print('covmat',self.covmat.shape)
        # print(self.covmat_inverse)

        nbins = int(self.nbins)
        # Read noise:
        self.cl_noise_matrix = self.ReadMatrix(self.cl_noise_file,self.cl_noise_order)

        # Read Chat and perhaps add noise:
        self.cl_hat_matrix = self.ReadMatrix(self.cl_hat_file,self.cl_hat_order)
        if not self.cl_hat_includes_noise:
            for k in range(nbins):
                self.cl_hat_matrix[k] += self.cl_noise_matrix[k]

        # Read cl_fiducial and perhaps add noise:
        self.cl_fiducial_sqrt_matrix = self.ReadMatrix(self.cl_fiducial_file,self.cl_fiducial_order)
        if not self.cl_fiducial_includes_noise:
            for k in range(nbins):
                self.cl_fiducial_sqrt_matrix[k] += self.cl_noise_matrix[k]
        # Now take matrix square root:
        for k in range(nbins):
            self.cl_fiducial_sqrt_matrix[k] = la.sqrtm(self.cl_fiducial_sqrt_matrix[k])


    def ReadMatrix(self, filename, crossmaps):
        """
        Read matrices for each ell-bin for all maps inside crossmaps and
        ordered in the same way as usedmaps. Returns list of matrices.

        """
        usedmaps = self.map_names_used.split()
        nmaps = len(usedmaps)
        # Get mask and indices
        indices, mask = self.GetIndicesAndMask(crossmaps.split())
        # Read matrix in packed format
        A = pd.read_table(os.path.join(self.data_directory, filename),comment='#',sep=' ',header=None, index_col=0).to_numpy()
        # Apply mask
        A = A[:,mask]

        # Create matrix for each bin and unpack A:
        Mlist = []
        # Loop over bins:
        for k in range(int(self.nbins)):
            M = np.zeros((nmaps,nmaps))
            Mflat = np.zeros(((nmaps*(nmaps+1))//2))
            Mflat[indices] = A[k,:]
            M[self.flat_to_diag] = Mflat
            # Symmetrise M and append to list:
            Mlist.append(M+M.T-np.diag(M.diagonal()))
        return Mlist

    def GetIndicesAndMask(self, crossmaplist):
        """
        Given a list of used maps and a list of available crossmaps, find a mask
        for the used crossmaps, and for each used crossmap, compute the falttened
        triangular index. We must allow map1 and map2 to be interchanged.
        If someone finds a nicer way to do this, please email me.
        """
        usedmaps = self.map_names_used.split()
        nmaps = len(usedmaps)
        mask = np.array([False for i in range(len(crossmaplist))])

        flatindex = []
        for i, crossmap in enumerate(crossmaplist):
            map1, map2 = crossmap.split('x')
            if map1 in usedmaps and map2 in usedmaps:
                index1 = usedmaps.index(map1)
                index2 = usedmaps.index(map2)
                # This calculates the flat index in a diagonal flattening:
                # if index1 > index2:
                #     flatindex.append((index1-index2)*(2*nmaps+1-index1+index2)/2+index2)
                # else:
                #     flatindex.append((index2-index1)*(2*nmaps+1-index2+index1)/2+index1)
                # This calculates the flat index in the standard numpy.tril_indices() way:
                if index1 > index2:
                    flatindex.append((index1*(index1+1))//2+index2)
                else:
                    flatindex.append((index2*(index2+1))//2+index1)
                mask[i] = True
        return flatindex, mask

    def ReadBandpasses(self):
        """
        Read bandpasses and compute some thermodynamic quantities.
        Everything stored in the dictionary self.bandpasses.
        """
        #Read bandpasses
        self.bandpasses = {}
        map_fields = self.map_fields.split()
        map_names = self.map_names.split()
        map_names_used = self.map_names_used.split()
        for key in map_names_used:
            self.bandpasses[key] = {'field':map_fields[map_names.index(key)],'filename':getattr(self, 'bandpass['+key+']')}

        for key, valdict in io_mp.dictitems(self.bandpasses):
            tmp = np.loadtxt(os.path.join(self.data_directory, valdict['filename']))
            #Frequency nu, response resp:
            valdict['nu'] = tmp[:,0]
            valdict['resp'] = tmp[:,1]
            valdict['dnu'] = np.gradient(valdict['nu'])

            # Calculate thermodynamic temperature conversion between this bandpass
            # and pivot frequencies 353 GHz (used for dust) and 23 GHz (used for
            # sync).
            th_int = np.sum(valdict['dnu']*valdict['resp']*valdict['nu']**4*np.exp(Ghz_Kelvin*valdict['nu']/T_CMB)/(np.exp(Ghz_Kelvin*valdict['nu']/T_CMB)-1.)**2)
            nu0=353.
            th0 = nu0**4*np.exp(Ghz_Kelvin*nu0/T_CMB) / (np.exp(Ghz_Kelvin*nu0/T_CMB) - 1.)**2
            valdict['th353'] = th_int / th0
            nu0=23.
            th0 = nu0**4*np.exp(Ghz_Kelvin*nu0/T_CMB) / (np.exp(Ghz_Kelvin*nu0/T_CMB) - 1.)**2
            valdict['th023'] = th_int / th0
            #print('th353:', valdict['th353'], 'th023:', valdict['th023'])
            # Calculate bandpass center-of-mass (i.e. mean frequency).
            valdict['nu_bar'] = np.sum(valdict['dnu']*valdict['resp']*valdict['nu'])/np.sum(valdict['dnu']*valdict['resp'])
            # Provide lambda function for computing bandpass center error given gamma_corr, gamma_95, gamma_150, gamma_220:
            if '95' in key:
                valdict['lambda_bandcenter_error'] = lambda gamma_corr, gamma_95, gamma_150, gamma_220: 1 + gamma_corr + gamma_95
            elif '150' in key:
                valdict['lambda_bandcenter_error'] = lambda gamma_corr, gamma_95, gamma_150, gamma_220: 1 + gamma_corr + gamma_150
            elif '220' in key:
                valdict['lambda_bandcenter_error'] = lambda gamma_corr, gamma_95, gamma_150, gamma_220: 1 + gamma_corr + gamma_220
            else:
                valdict['lambda_bandcenter_error'] = lambda gamma_corr, gamma_95, gamma_150, gamma_220: 1

    def logprior(self, cosmo, data):
        BBbetadust = data.mcmc_parameters['BBbetadust']['current']*data.mcmc_parameters['BBbetadust']['scale']
        BBbetasync = data.mcmc_parameters['BBbetasync']['current']*data.mcmc_parameters['BBbetasync']['scale']
        loglkl_prior = -0.5 * self.use_beta_dust_prior*(BBbetadust - self.mean_BBbetadust) ** 2 / (self.sigma_BBbetadust ** 2) -0.5 * (BBbetasync - self.mean_BBbetasync) ** 2 / (self.sigma_BBbetasync ** 2)
        return loglkl_prior

    def loglkl(self, cosmo, data):
        """
        Compute negative log-likelihood using the Hamimeche-Lewis formalism, see
        http://arxiv.org/abs/arXiv:0801.0554
        """
        # Define the matrix transform
        def MatrixTransform(C, Chat, CfHalf):
            # C is real and symmetric, so we can use eigh()
            D, U = la.eigh(C)
            D = np.abs(D)
            S = np.sqrt(D)
            # Now form B = C^{-1/2} Chat C^{-1/2}. I am using broadcasting to divide rows and columns
            # by the eigenvalues, not sure if it is faster to form the matmul(S.T, S) matrix.
            # B = U S^{-1} V^T Chat U S^{-1} U^T
            B = np.dot(np.dot(U,np.dot(np.dot(U.T,Chat),U)/S[:,None]/S[None,:]),U.T)
            # Now evaluate the matrix function g[B]:
            D, U = la.eigh(B)
            gD = np.sign(D-1.)*np.sqrt(2.*np.maximum(0.,D-np.log(D)-1.))
            # Final transformation. U*gD = U*gD[None,:] done by broadcasting. Collect chain matrix multiplication using reduce.
            M = functools.reduce(np.dot, [CfHalf,U*gD[None,:],U.T,CfHalf.T])
            #M = np.dot(np.dot(np.dot(CfHalf,U*gD[None,:]),U.T),Cfhalf.T)
            return M

        # Recover Cl_s from CLASS, which is a dictionary, with the method
        # get_cl from the Likelihood class, because it already makes the
        # conversion to uK^2.
        dict_Cls = self.get_cl(cosmo, self.cl_lmax)
        # Make short hand expressions and remove l=0.
        ell = dict_Cls['ell'][1:]
        DlEE = ell*(ell+1)*dict_Cls['ee'][1:]/(2*np.pi)
        DlBB = ell*(ell+1)*dict_Cls['bb'][1:]/(2*np.pi)
        # Update foreground model
        self.UpdateForegroundModel(cosmo, data)
        #Make names and fields into lists
        map_names = self.map_names_used.split()
        map_fields = self.map_fields_used
        nmaps = len(map_names)
        ncrossmaps = nmaps*(nmaps+1)//2
        nbins = int(self.nbins)
        # Initialise Cls matrix to zero:
        Cls = np.zeros((nbins,nmaps,nmaps))
        # Initialise the X vector:
        X = np.zeros((nbins*ncrossmaps))
        for i in range(nmaps):
            for j in range(i+1):
                #If EE or BB, add theoretical prediction including foreground:
                if map_fields[i]==map_fields[j]=='E' or map_fields[i]==map_fields[j]=='B':
                    map1 = map_names[i]
                    map2 = map_names[j]
                    dust = self.fdust[map1]*self.fdust[map2]
                    sync = self.fsync[map1]*self.fsync[map2]
                    dustsync = self.fdust[map1]*self.fsync[map2] + self.fdust[map2]*self.fsync[map1]
                    # if EE spectrum, multiply foregrounds by the EE/BB ratio:
                    if map_fields[i]=='E':
                        dust = dust * self.EEtoBB_dust
                        sync = sync * self.EEtoBB_sync
                        dustsync = dustsync * np.sqrt(self.EEtoBB_dust*self.EEtoBB_sync)
                        # Deep copy is important here, since we want to reuse DlXX for each map.
                        DlXXwithforegound = np.copy(DlEE)
                    else:
                        DlXXwithforegound = np.copy(DlBB)
                    # Finally add the foreground model:
                    DlXXwithforegound += (dust*self.dustcoeff*self.Deltap_dust[(i, j)] +
                                          sync*self.synccoeff*self.Deltap_sync[(i, j)] +
                                          dustsync*self.dustsynccoeff)
                    # Apply the binning using the window function:
                    for k in range(nbins):
                        Cls[k,i,j] = Cls[k,j,i] = np.dot(DlXXwithforegound,self.window_data[k,:,self.diag_to_flat[i,j]])
        # Add noise contribution:
        for k in range(nbins):
            Cls[k,:,:] += self.cl_noise_matrix[k]
            # Compute entries in X vector using the matrix transform
            T = MatrixTransform(Cls[k,:,:], self.cl_hat_matrix[k], self.cl_fiducial_sqrt_matrix[k])
            # Add flat version of T to the X vector
            X[k*ncrossmaps:(k+1)*ncrossmaps] = T[self.flat_to_diag]
        # Compute chi squared
        chi2 = np.dot(X.T,np.dot(self.covmat_inverse,X))
        # Compute prior
        loglkl_prior = self.logprior(cosmo, data)
        return -0.5*chi2 + loglkl_prior


    def UpdateForegroundModel(self, cosmo, data):
        """
        Update the foreground model.
        """
        # Function to compute f_dust
        def DustScaling(beta, Tdust, bandpass):
            # Calculates greybody scaling of dust signal defined at 353 GHz to specified bandpass.
            nu0 = 353 #Pivot frequency for dust (353 GHz).
            # Integrate greybody scaling and thermodynamic temperature conversion across experimental bandpass.
            gb_int = np.sum(bandpass['dnu']*bandpass['resp']*bandpass['nu']**(3+beta)/(np.exp(Ghz_Kelvin*bandpass['nu']/Tdust) - 1))
            # Calculate values at pivot frequency.
            gb0 = nu0**(3+beta) / (np.exp(Ghz_Kelvin*nu0/Tdust) - 1)
            # Retrieve band center error
            bandcenter_err = bandpass['bandcenter_error']
            # Calculate and return dust scaling fdust.
            if bandcenter_err == 1:
                return ((gb_int / gb0) / bandpass['th353'])
            else:
                th_err = (bandcenter_err**4 * np.exp(Ghz_Kelvin * bandpass['nu_bar'] * (bandcenter_err - 1) / T_CMB) * 
                           (np.exp(Ghz_Kelvin * bandpass['nu_bar'] / T_CMB) - 1)**2 / (np.exp(Ghz_Kelvin * bandpass['nu_bar'] * bandcenter_err / T_CMB) - 1)**2)
                gb_err = (bandcenter_err**(3+beta) * (np.exp(Ghz_Kelvin * bandpass['nu_bar'] / Tdust) - 1) / 
                          (np.exp(Ghz_Kelvin * bandpass['nu_bar'] * bandcenter_err / Tdust) - 1))
                return ((gb_int / gb0) / bandpass['th353'] * (gb_err / th_err))

        # Function to compute f_sync
        def SyncScaling(beta, bandpass):
            #Calculates power-law scaling of synchrotron signal defined at 150 GHz to specified bandpass.
            nu0 = 23.0 # Pivot frequency for sync (23 GHz).
            # Integrate power-law scaling and thermodynamic temperature conversion across experimental bandpass.
            pl_int = np.sum( bandpass['dnu']*bandpass['resp']*bandpass['nu']**(2+beta))
            # Calculate values at pivot frequency.
            pl0 = nu0**(2+beta)
            # Retrieve band center error
            bandcenter_err = bandpass['bandcenter_error']
            # Calculate and return dust scaling fsync.
            if bandcenter_err == 1:
                return ((pl_int / pl0) / bandpass['th023'])
            else:
                th_err = (bandcenter_err**4 * np.exp(Ghz_Kelvin * bandpass['nu_bar'] * (bandcenter_err - 1) / T_CMB) * 
                          (np.exp(Ghz_Kelvin * bandpass['nu_bar'] / T_CMB) - 1)**2 / 
                          (np.exp(Ghz_Kelvin * bandpass['nu_bar'] * bandcenter_err / T_CMB) - 1)**2)
                pl_err = (bandcenter_err)**(2+beta)
                return ((pl_int / pl0) / bandpass['th023'] * (pl_err / th_err))
            
        # Function to calculate the factor by which foreground (dust or sync) power is decreased
        # for a cross-spectrum between two different frequencies.
        def Decorrelation(ell, bandpass1, bandpass2, dust_or_sync):
            l_pivot = 80.0
            nu0 = bandpass1['nu_bar']*bandpass1['bandcenter_err']
            nu1 = bandpass2['nu_bar']*bandpass2['bandcenter_err']
            # Assign pivot frequencies for foreground decorrelation model.
            if dust_or_sync == 'dust':
                Delta = self.Delta_dust
                nu0_pivot = 217.0
                nu1_pivot = 353.0
                lform = self.lform_dust_decorr
            elif dust_or_sync == 'sync':
                Delta = self.Delta_sync
                nu0_pivot = 23.0
                nu1_pivot = 33.0
                lform = self.lform_sync_decorr
            else:
                raise ValueError('Unknown foreground type: %s' % dust_or_sync)
            # Decorrelation scales as log^2(nu0/nu1)
            scl_nu = (np.log(nu0 / nu1)**2) / (np.log(nu0_pivot / nu1_pivot)**2)
            # Functional form for ell scaling is specified in .dataset file.
            scl_ell = {'flat': np.ones(len(ell)), 'lin': ell / l_pivot, 'quad': (ell / l_pivot)**2}[lform]

            # Even for small cval, correlation can become negative for sufficiently large frequency
            # difference or ell value (with linear or quadratic scaling).
            # Following Vansyngel et al, A&A, 603, A62 (2017), we use an exponential function to
            # remap the correlation coefficient on to the range [0,1].
            # We symmetrically extend this function to (non-physical) correlation coefficients
            # greater than 1 -- this is only used for validation tests of the likelihood model.
            # Value returned corresponds to the "re-mapped" decorrelation parameter, denoted as
            # $\Delta'_d$ in Appendix F of the BK15 paper (equations F4 and F5)
            if (Delta > 1):
                # If using a physical prior for Delta, then this scenario should never happen.
                Deltap = 2.0 - np.exp(np.log(2.0 - Delta) * scl_nu * scl_ell)
            else:
                # This is for physically-relevant values of Delta.
                Deltap = np.exp(np.log(Delta) * scl_nu * scl_ell)
            return Deltap
        
        ellpivot = 80.
        ell = np.arange(1,int(self.cl_lmax)+1)

        # Convenience variables: store the nuisance parameters in short named variables
        # for parname in self.use_nuisance:
        #     evalstring = parname+" = data.mcmc_parameters['"+parname+"']['current']*data.mcmc_parameters['"+parname+"']['scale']"
        #     print(evalstring)
        BBdust = data.mcmc_parameters['BBdust']['current']*data.mcmc_parameters['BBdust']['scale']
        BBsync = data.mcmc_parameters['BBsync']['current']*data.mcmc_parameters['BBsync']['scale']
        BBalphadust = data.mcmc_parameters['BBalphadust']['current']*data.mcmc_parameters['BBalphadust']['scale']
        BBbetadust = data.mcmc_parameters['BBbetadust']['current']*data.mcmc_parameters['BBbetadust']['scale']
        BBTdust = data.mcmc_parameters['BBTdust']['current']*data.mcmc_parameters['BBTdust']['scale']
        BBalphasync = data.mcmc_parameters['BBalphasync']['current']*data.mcmc_parameters['BBalphasync']['scale']
        BBbetasync = data.mcmc_parameters['BBbetasync']['current']*data.mcmc_parameters['BBbetasync']['scale']
        BBdustsynccorr = data.mcmc_parameters['BBdustsynccorr']['current']*data.mcmc_parameters['BBdustsynccorr']['scale']

        # Store current EEtoBB conversion parameters.
        self.EEtoBB_dust = data.mcmc_parameters['EEtoBB_dust']['current']*data.mcmc_parameters['EEtoBB_dust']['scale']
        self.EEtoBB_sync = data.mcmc_parameters['EEtoBB_sync']['current']*data.mcmc_parameters['EEtoBB_sync']['scale']

        # Store current dust correlation ratio between 217 and 353 GHz, ell=80
        self.Delta_dust = data.mcmc_parameters['Delta_dust']['current']*data.mcmc_parameters['Delta_dust']['scale']
        # Store sync correlation ratio between 23 and 33 GHz, ell=80
        self.Delta_sync = data.mcmc_parameters['Delta_sync']['current']*data.mcmc_parameters['Delta_sync']['scale']

        # Update band center error from nuisance parameters:
        gamma_corr = data.mcmc_parameters['gamma_corr']['current']*data.mcmc_parameters['gamma_corr']['scale']
        gamma_95 = data.mcmc_parameters['gamma_95']['current']*data.mcmc_parameters['gamma_95']['scale']
        gamma_150 = data.mcmc_parameters['gamma_150']['current']*data.mcmc_parameters['gamma_150']['scale']
        gamma_220 = data.mcmc_parameters['gamma_220']['current']*data.mcmc_parameters['gamma_220']['scale']
        for key, bandpass in io_mp.dictitems(self.bandpasses):
            bandpass['bandcenter_error'] = bandpass['lambda_bandcenter_error'](gamma_corr, gamma_95, gamma_150, gamma_220)

        # Compute fdust and fsync for each bandpass
        self.fdust = {}
        self.fsync = {}
        for key, bandpass in io_mp.dictitems(self.bandpasses):
            self.fdust[key] = DustScaling(BBbetadust, BBTdust, bandpass)
            self.fsync[key] = SyncScaling(BBbetasync, bandpass)

        # Computes coefficients such that the foreground model is simply
        # dust*self.dustcoeff*self.Deltap_dust[(i, j)]+sync*self.synccoeff*self.Deltap_sync[(i, j)]+dustsync*self.dustsynccoeff
        self.dustcoeff = BBdust*(ell/ellpivot)**BBalphadust
        self.synccoeff = BBsync*(ell/ellpivot)**BBalphasync
        self.dustsynccoeff = BBdustsynccorr*np.sqrt(BBdust*BBsync)*(ell/ellpivot)**(0.5*(BBalphadust+BBalphasync))
        self.Deltap_dust = defaultdict(lambda: 1.0)
        self.Deltap_sync = defaultdict(lambda: 1.0)
        map_names_used = self.map_names_used.split()
        for i, map1 in enumerate(map_names_used):
            for j in range(i): # We take i and not 1 + j because there is no decorrelation for the same map.
                map2 = map_names_used[j]
                if abs(self.Delta_dust - 1.0) > 1e-5:
                    self.Deltap_dust[(i, j)] = Decorrelation(ell, self.bandpasses[map1], self.bandpasses[map2], 'dust')
                if abs(self.Delta_sync - 1.0) > 1e-5:
                    self.Deltap_sync[(i, j)] = Decorrelation(ell, self.bandpasses[map1], self.bandpasses[map2], 'sync')
