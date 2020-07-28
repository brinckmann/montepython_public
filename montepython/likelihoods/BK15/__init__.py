"""
.. module:: BK15
    :synopsis: BK15 likelihood

.. moduleauthor:: Deanna C. Hooper <deanna.hooper@ulb.ac.be>, adapted from the BK14 version (Thomas Tram <thomas.tram@port.ac.uk>)
Last updated Mar 25, 2020. Based on the Public Release of the BK2015 data (1810.05216) and the corresponding CosmoMC module.
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
import montepython.io_mp as io_mp
import os
from functools import reduce
from montepython.likelihood_class import Likelihood_sn

T_CMB = 2.72548        # CMB temperature
h = 6.62606957e-34     # Planck's constant
kB = 1.3806488e-23     # Boltzmann constant
Ghz_Kelvin = h/kB*1e9  # GHz Kelvin conversion

class BK15(Likelihood_sn):

    def __init__(self, path, data, command_line):
        # Unusual construction, since the data files are not distributed
        # alongside BK15 (size problems)
        try:
            # Read the .dataset file specifying the data.
            super(BK15, self).__init__(path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The BK15 data files were not found. Please download the "
                "data from http://bicepkeck.org/bk15_2018_release.html"
                ", extract it, and copy the BK15 folder inside"
                "`BK15_cosmomc/data/` to `your_montepython/data/`")

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

        # Read window bins
        self.window_data = np.zeros((int(self.nbins),int(self.cl_lmax),ncrossmaps))
        # Retrieve mask and index permutation of windows:
        indices, mask = self.GetIndicesAndMask(self.bin_window_in_order.split())
        for k in range(nbins):
            windowfile = os.path.join(self.data_directory, self.bin_window_files.replace('%u',str(k+1)))
            #tmp = pd.read_table(windowfile,comment='#',sep=' ',header=None, index_col=0).to_numpy()
            tmp = pd.read_csv(windowfile,comment='#',sep=' ',header=None, index_col=0).values
            # Apply mask
            tmp = tmp[:,mask]
            # Permute columns and store this bin
            self.window_data[k][:,indices] = tmp
        # print 'window_data',self.window_data.shape

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

        tmp = pd.read_csv(os.path.join(self.data_directory, self.covmat_fiducial),comment='#',sep=' ',header=None,skipinitialspace=True).values
        # Apply mask:
        tmp = tmp[:,supermask][supermask,:]
        print('Covmat read with shape {}'.format(tmp.shape))
        # Store covmat in correct order
        self.covmat = np.zeros((nbins*ncrossmaps,nbins*ncrossmaps))
        for index_tmp, index_covmat in enumerate(superindices):
            self.covmat[index_covmat,superindices] = tmp[index_tmp,:]

        #Compute inverse and store
        self.covmat_inverse = la.inv(self.covmat)
        # print 'covmat',self.covmat.shape
        # print self.covmat_inverse

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
        A = pd.read_csv(os.path.join(self.data_directory, filename),comment='#',sep=' ',header=None, index_col=0).values
        # Apply mask
        A = A[:,mask]

        # Create matrix for each bin and unpack A:
        Mlist = []
        # Loop over bins:
        for k in range(int(self.nbins)):
            M = np.zeros((nmaps,nmaps))
            Mflat = np.zeros((nmaps*(nmaps+1)//2))
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
                    flatindex.append(index1*(index1+1)//2+index2)
                else:
                    flatindex.append(index2*(index2+1)//2+index1)
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

        for key, valdict in self.bandpasses.items():
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
            #print 'th353:', valdict['th353'], 'th023:', valdict['th023']


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
            M = reduce(np.dot, [CfHalf,U*gD[None,:],U.T,CfHalf.T])
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
                    DlXXwithforegound += (dust*self.dustcoeff+sync*self.synccoeff+dustsync*self.dustsynccoeff)
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
        return -0.5*chi2


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
            # Calculate and return dust scaling fdust.
            return ((gb_int / gb0) / bandpass['th353'])

        # Function to compute f_sync
        def SyncScaling(beta, bandpass):
            #Calculates power-law scaling of synchrotron signal defined at 150 GHz to specified bandpass.
            nu0 = 23.0 # Pivot frequency for sync (23 GHz).
            # Integrate power-law scaling and thermodynamic temperature conversion across experimental bandpass.
            pl_int = np.sum( bandpass['dnu']*bandpass['resp']*bandpass['nu']**(2+beta))
            # Calculate values at pivot frequency.
            pl0 = nu0**(2+beta)
            # Calculate and return dust scaling fsync.
            return ((pl_int / pl0) / bandpass['th023'])


        ellpivot = 80.
        ell = np.arange(1,int(self.cl_lmax)+1)

        # Convenience variables: store the nuisance parameters in short named variables
        # for parname in self.use_nuisance:
        #     evalstring = parname+" = data.mcmc_parameters['"+parname+"']['current']*data.mcmc_parameters['"+parname+"']['scale']"
        #     print evalstring
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

        # Compute fdust and fsync for each bandpass
        self.fdust = {}
        self.fsync = {}
        for key, bandpass in self.bandpasses.items():
            self.fdust[key] = DustScaling(BBbetadust, BBTdust, bandpass)
            self.fsync[key] = SyncScaling(BBbetasync, bandpass)

        # Computes coefficients such that the foreground model is simply
        # dust*self.dustcoeff+sync*self.synccoeff+dustsync*self.dustsynccoeff
        # These coefficients are independent of the map used,
        # so we save some time by computing them here.
        self.dustcoeff = BBdust*(ell/ellpivot)**BBalphadust
        self.synccoeff = BBsync*(ell/ellpivot)**BBalphasync
        self.dustsynccoeff = BBdustsynccorr*np.sqrt(BBdust*BBsync)*(ell/ellpivot)**(0.5*(BBalphadust+BBalphasync))
