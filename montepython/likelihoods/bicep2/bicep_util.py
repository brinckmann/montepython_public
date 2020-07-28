# bicep_util.py
#
# This is a module containing subfunctions to evaluate the bicep1 or bicep2 likelihood
#
# get_bpwf
# load_cmbfast
# calc_expvals
# read_data_products_bandpowers
# read_M
# calc_vecp
# g
# vecp
# saveLikelihoodToText
#
#$Id: bicep_util.py,v 1.1.2.5 2014/03/12 18:20:57 dbarkats Exp $ #

# IMPORTANT NOTE: This version was modified by Benjamin Audren, in order to be
# flexible enough to work with a slightly different configuration.
from __future__ import print_function
import os
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
import io_mp

import sys
if not sys.version_info[0] < 3:
    from past.builtins import xrange

#####################################################################
def get_bpwf(exp='bicep1', root=''):
    # This assumes you have the files
    # windows/B1_3yr_bpwf_bin[1-9]_20131003.txt in the root directory
    # windows/B2_3yr_bpwf_bin[1-9]_date.txt in the working directory if exp = 'bicep2'

    if exp == 'bicep1':
        # Load up BICEP1 bandpower window functions
        file_in = os.path.join("windows", "B1_3yr_bpwf_bin?_20131003.txt")
        print("### Reading the BICEP1 BPWF from  file: %s" % file_in)
        ncol = 580
    elif exp == 'bicep2':
        # Load up BICEP2 bandpower window functions
        file_in = os.path.join("windows", "B2_3yr_bpwf_bin?_20140314.txt")
        print("### Reading the BICEP2 BPWF from  file: %s" % file_in)
        ncol = 599
    else:
        print('exp must be "bicep1" or "bicep2" to load the proper window functions')
        print('window functions must be in the root_directory/windows/')
        print('bicep2 window functions available at http://bicepkeck.org/bicep2_2014_release')
        print('bicep1 window functions available at bicep.rc.fas.harvard.edu/bicep1_3yr')
        raise OSError()

    # Initialize array so it's just like our Matlab version
    bpwf_Cs_l = np.zeros([ncol, 9, 6])

    for i in xrange(9):
        window_file = file_in.replace('?', str(i+1))

        try:
            data = np.loadtxt(
                os.path.join(root, window_file))
        except OSError:
            print("Error reading  %s." % window_file +
                  "Make sure it is in root directory")
            raise OSError()
        bpwf_Cs_l[:, i, 0] = data[:, 1]   # TT -> TT
        bpwf_Cs_l[:, i, 1] = data[:, 2]   # TE -> TE
        bpwf_Cs_l[:, i, 2] = data[:, 3]   # EE -> EE
        bpwf_Cs_l[:, i, 3] = data[:, 4]   # BB -> BB

    bpwf_l = data[:, 0]

    return (bpwf_l, bpwf_Cs_l)

#####################################################################
def load_cmbfast(file_in):
    # Equivalent of load_cmbfast.m but doesn't read .fits for now
    # (when it does, may want to change back).  Right now we just want
    # a simple .txt spectrum with columns
    # We want the columns ordered TT TE EE BB TB EB.  Note
    # that standard CAMB output is TT EE BB TE...
    # TB, EB, BT, BE are already zero.

    print("### Loading input spectra from file: %s" % file_in)

    try:
        data = np.loadtxt(file_in)
    except:
        print("Error reading %s. Make sure it is in working directory" %file_in)
    ell = data[:, 0]

    # Initialize the Cs_l array
    Cs_l = np.zeros([np.shape(data)[0], 9])

    Cs_l[:, 0] = data[:, 1]  # TT
    Cs_l[:, 1] = data[:, 2]  # TE
    Cs_l[:, 2] = data[:, 3]  # EE
    Cs_l[:, 3] = data[:, 4]  # BB
    # Cs_l[:,4]               # TB
    # Cs_l[:,5]               # EB
    Cs_l[:, 6] = data[:, 2]  # ET
    # Cs_l[:,7]             # BT
    # Cs_l[:,8] =           # BE

    return (ell, Cs_l)

#####################################################################
def calc_expvals(inpmod_l, inpmod_Cs_l, bpwf_l, bpwf_Cs_l):

    # Inputs
    #         inpmod: theory spectrum loaded by load_cmbfast (l, Cs_l)
    #                 Contents: TT, TE, EE, BB, TB, EB, ET, BT, BE
    #         bpwf: bandpower window function from reduc_bpwf (l, Cs_l)
    #                 Contents: TT, TP, EE->EE, BB->BB, EE->BB, BB->EE

    nbin = np.shape(bpwf_Cs_l)[1]

    # Don't assume inpmod and bpwf start at the same ell --
    # CAMB spectra like to start at l=0 but bpwf can be higher.
    # We do assume that both have delta ell = 1
    nl = np.shape(bpwf_Cs_l)[0]
    indx = np.arange(0,nl)   # Python ranges want one more...
    indx = indx + np.nonzero(bpwf_l[0]==inpmod_l)[0][0]  # don't subtract 1

    # Initialize expval array
    expv = np.zeros([nbin,np.shape(bpwf_Cs_l)[2]])

    # TT
    x = bpwf_Cs_l[:,:,0]*np.transpose(np.tile(inpmod_Cs_l[indx,0],(nbin,1)))
    expv[:,0] = np.sum(x,0)

    # TE
    x = bpwf_Cs_l[:,:,1]*np.transpose(np.tile(inpmod_Cs_l[indx,1],(nbin,1)))
    expv[:,1] = np.sum(x,0)

    # EE: x1 = EE->EE, x2 = BB->EE
    x1 = bpwf_Cs_l[:,:,2]*np.transpose(np.tile(inpmod_Cs_l[indx,2],(nbin,1)))
    x2 = bpwf_Cs_l[:,:,5]*np.transpose(np.tile(inpmod_Cs_l[indx,3],(nbin,1)))
    expv[:,2] = np.sum(x1,0) + np.sum(x2,0)

    # BB: x1 = BB->BB, x2 = EE->BB
    x1 = bpwf_Cs_l[:,:,3]*np.transpose(np.tile(inpmod_Cs_l[indx,3],(nbin,1)))
    x2 = bpwf_Cs_l[:,:,4]*np.transpose(np.tile(inpmod_Cs_l[indx,2],(nbin,1)))
    expv[:,3] = np.sum(x1,0) + np.sum(x2,0)

    # expv of TB, EB zero as initialized

    return expv


#####################################################################
# Loads matrices C_fl: fiducial bandpowers (mean of s+n sims).
#             C_l_hat: real data bandpowers
#             and N_l: Noise bias bandpowers
#  outputs them in an array bandpowers[i][j]
# i=0,1,2 for the three bandpower matrices j=0..8 for the 9 'l' bins

def read_data_products_bandpowers(exp='bicep1', root=""):

    if exp == 'bicep1':
        file_in="B1_3yr_likelihood_bandpowers_20131003.txt"
    elif exp == 'bicep2':
        file_in="B2_3yr_likelihood_bandpowers_20140314.txt"
    else:
        print('exp must be "bicep1" or "bicep2" to load the proper files')

    print("### Reading fiducial, real, and noise bias bandpowers from file: %s"%file_in)

    values = list()
    try:
        fin = open(os.path.join(root, file_in), 'r')
    except OSError:
        print("Error reading %s. Make sure it is in root directory" %file_in)
    for line in fin:
        if "#" not in line:
            lst = line.split(' ')
            if len(lst) > 3:
                b = []
                for elem in lst:
                    if elem != '':
                        b.append( float( elem ) )
                values.append(b)

    bandpowers = []
    for i in range(3):
        c = list()
        for j in range(9):
            c.append(values[ i*27 + j * 3: i*27 + j * 3 + 3 ])
        bandpowers.append(c)

    return bandpowers


#####################################################################
# Loads the M_cc matrix
# for bicep1 see details as defined in Barkats et al section 9.1

def read_M(exp='bicep1', root=""):

    if exp =='bicep1':
        file_in = "B1_3yr_bpcm_20131003.txt"
    elif exp == 'bicep2':
        file_in = "B2_3yr_bpcm_no-sysuncer_20140314.txt"
    else:
        print('exp must be bicep1 or bicep2 to load the proper files')

    print("### Reading covariance matrix (M_cc) from file: %s" %file_in)

    try:
        data = np.loadtxt(os.path.join(root, file_in))
    except OSError:
        print("Error reading %s. Make sure it is in working directory" %file_in)

    # HACK because file_in = "B2_3yr_bpcm_no-sysuncer_20140226.txt"  has different format
    if exp == 'bicep2':
        data = data.reshape((54,54))
    M_raw = np.array(data)
    return M_raw


#####################################################################
# Utility functions used to calculate the likelihood
# for a given l bin.

def calc_vecp(l,C_l_hat,C_fl, C_l):

    C_fl_12 = sqrtm(C_fl[l])
    C_l_inv = LA.inv(C_l[l])
    C_l_inv_12= sqrtm(C_l_inv)
    # the order is inverted compared to matlab hamimeche_lewis_likelihood.m line 19

    # line 20 of hamimeche_lewis_likelihood.m
    res = np.dot(C_l_inv_12, np.dot(C_l_hat[l], C_l_inv_12))
    [d, u] = LA.eigh(res)
    d = np.diag(d)  # noticed that python returns the eigenvalues as a vector, not a matrix
    #np. dot( u, np.dot( np.diag(d), LA.inv(u))) should be equals to res
    # real symmetric matrices are diagnalized by orthogonal matrices (M^t M = 1)

    # this makes a diagonal matrix by applying g(x) to the eigenvalues, equation 10 in Barkats et al
    gd = np.sign(np.diag(d) - 1) * np.sqrt(2 * (np.diag(d) - np.log(np.diag(d)) - 1))
    gd = np.diag(gd)
    # Argument of vecp in equation 8; multiplying from right to left
    X = np.dot(np.transpose(u), C_fl_12)
    X = np.dot(gd, X)
    X = np.dot(u, X)
    X = np.dot(C_fl_12, X)
    # This is the vector of equation 7
    X = vecp(X)

    return X


#def g(x):
#    #  sign(x-1) \sqrt{ 2(x-ln(x) -1 }
#    return np.sign(x-1) * np.sqrt( 2* (x - np.log(x) -1) )

def vecp(mat):
    # This returns the unique elements of a symmetric matrix
    # 2014-02-11 now mirrors matlab vecp.m

    dim = mat.shape[0]

    vec = np.zeros((dim*(dim+1)//2))
    counter = 0
    for iDiag in range(0,dim):
        vec[counter:counter+dim-iDiag] = np.diag(mat,iDiag)

        counter = counter + dim - iDiag

    return vec

#####################################################################
# Function to evaluate the likelihood itself
def evaluateLikelihood(C_l,C_l_hat,C_fl,M_inv):
    logL = 0
    # Calculate X vector (Eq 8) for each l, lp
    for l in range(0,9):
        X = calc_vecp(l,C_l_hat,C_fl,C_l)
        for lp in range(0,9):
            #print(l, lp, r)
            Xp = calc_vecp(lp,C_l_hat,C_fl,C_l)
            M_inv_pp = M_inv[l,lp,:,:]
            # calculate loglikelihood (Eq 7)
            thislogL = (-0.5)*np.dot(X,np.dot(M_inv_pp,Xp))
            logL = logL + thislogL

    if np.isnan(logL):
        logL = -1e20

    logL = np.real(logL)
    return logL

#####################################################################
# Utility function  to save the likelihood vs r in a text file

def saveLikelihoodToText(rlist, logLike, field, exp='bicep1'):

    if exp == 'bicep1':
        print("### Saving Likelihood to file: B1_logLike.txt...")
        f = open("B1_logLike.txt", "w")
        f.write('# BICEP1 likelihood for r \n')
        f.write('# Based on data from:  Barkats et al, Degree Scale CMB Polarization Measurements from Three Years of BICEP1 Data \n')
        f.write('# Available at http://bicep.rc.fas.harvard.edu/bicep1_3yr/ \n')
        f.write('# This text file contains the tabulated likelihood for the tensor-to-scalar ratio, r, derived from the BICEP1 %s spectrum. \n'%field)
        f.write('# Calculated via the "Hamimeche-Lewis likelihood" method described in Section 9.1 of Barkats et al. \n')
        f.write('# This file is generated from a standalone python module: b1_r_wrapper.py \n')
        f.write('# This likelihood curve corresponds to the blue curve from the left-hand panel of Figure 10 from Barkats et al. \n')
        f.write('# \n')
        f.write('# Columns:  r, logLiklelihood(r) \n')
    elif exp == 'bicep2':
        print("### Saving Likelihood to file: B2_logLike.txt...")
        f = open("B2_logLike.txt", "w")
        f.write('# BICEP2 likelihood\n')
        f.write('# Based on data from: DETECTION OF B-mode POLARIZATION AT DEGREE SCALES USING BICEP2 \n')
        f.write('# Available at  http://www.bicepkeck.org/bicep2_2014_release/ \n')
        f.write('# This text file contains the tabulated likelihood derived from the BICEP2 %s spectrum. \n'%field)
        f.write('# Calculated via the "Hamimeche-Lewis likelihood" method described in Sectiox [specify seciotn and paper title/author here.] \n')
        f.write('# This file is generated from a standalone python module: bicep_r_wrapper.py \n')
        f.write('# \n')
        f.write('# Columns:  r, logLiklelihood(r) \n')

    for i in range(0,len(rlist)):
        f.write('%6.3f %6.4e \n'%(rlist[i],logLike[i]))
    f.close()


#####################################################################
#  This function loads:
#   - the bandpower data products (C_fl, C_l_hat, N_l),
#   - the covariance matrix and processes it to output the inverse
#   - the bandpower window functions
#
def init(experiment, field, root=""):
    """
    Initialize all quantities for likelihood computation

    KeyWord Arguments
    -----------------
    root: str
        specify the working directory to explore

    """

    # load the bandpower window functions
    (bpwf_l,bpwf_Cs_l) = get_bpwf(exp=experiment, root=root)

    # load the  bandpower products
    bp = read_data_products_bandpowers(exp=experiment, root=root)
    bp = np.array(bp)

    # initialize bandpower arrays
    nf = len(field)
    dim = nf*(nf+1)//2
    C_l_hat = np.zeros((9, nf, nf))
    C_fl = np.zeros((9, nf, nf))
    N_l = np.zeros((9, nf, nf))
    C_l = np.zeros((9, nf, nf))

    #Selects parts of the necessary matrices for a given instance of the field
    if field == "T":
        C_l_hat[:, 0, 0] = bp[1, :, 0, 0]
        C_fl[:, 0, 0] = bp[0, :, 0, 0]
        N_l[:, 0, 0] = bp[2, :, 0, 0]
    elif field == "E":
        C_l_hat[:, 0, 0] = bp[1, :, 1, 1]
        C_fl[:, 0, 0] = bp[0, :, 1, 1]
        N_l[:, 0, 0] = bp[2, :, 1, 1]
    elif field == "B":
        C_l_hat[:, 0, 0] = bp[1, :, 2, 2]
        C_fl[:, 0, 0] = bp[0, :, 2, 2]
        N_l[:, 0, 0] = bp[2, :, 2, 2]
    elif field == "EB":
        C_l_hat[:, 0, 0] = bp[1, :, 1, 1]  # EE
        C_l_hat[:, 0, 1] = bp[1, :, 1, 2]  # EB
        C_l_hat[:, 1, 0] = bp[1, :, 2, 1]  # BE
        C_l_hat[:, 1, 1] = bp[1, :, 2, 2]  # BB
        C_fl[:, 0, 0] = bp[0, :, 1, 1]
        C_fl[:, 0, 1] = bp[0, :, 1, 2]
        C_fl[:, 1, 0] = bp[0, :, 2, 1]
        C_fl[:, 1, 1] = bp[0, :, 2, 2]
        N_l[:, 0, 0] = bp[2, :, 1, 1]
        N_l[:, 0, 1] = bp[2, :, 1, 2]
        N_l[:, 1, 0] = bp[2, :, 2, 1]
        N_l[:, 1, 1] = bp[2, :, 2, 2]
    elif field == "TB":
        C_l_hat[:, 0, 0] = bp[1, :, 0, 0]  # TT
        C_l_hat[:, 0, 1] = bp[1, :, 0, 2]  # TB
        C_l_hat[:, 1, 0] = bp[1, :, 2, 0]  # BT
        C_l_hat[:, 1, 1] = bp[1, :, 2, 2]  # BB
        C_fl[:, 0, 0] = bp[0, :, 0, 0]
        C_fl[:, 0, 1] = bp[0, :, 0, 2]
        C_fl[:, 1, 0] = bp[0, :, 2, 0]
        C_fl[:, 1, 1] = bp[0, :, 2, 2]
        N_l[:, 0, 0] = bp[2, :, 0, 0]
        N_l[:, 0, 1] = bp[2, :, 0, 2]
        N_l[:, 1, 0] = bp[2, :, 2, 0]
        N_l[:, 1, 1] = bp[2, :, 2, 2]
    elif field == "TE":
        C_l_hat[:, 0, 0] = bp[1, :, 0, 0]  # TT
        C_l_hat[:, 0, 1] = bp[1, :, 0, 1]  # TE
        C_l_hat[:, 1, 0] = bp[1, :, 1, 0]  # ET
        C_l_hat[:, 1, 1] = bp[1, :, 1, 1]  # EE
        C_fl[:, 0, 0] = bp[0, :, 0, 0]
        C_fl[:, 0, 1] = bp[0, :, 0, 1]
        C_fl[:, 1, 0] = bp[0, :, 1, 0]
        C_fl[:, 1, 1] = bp[0, :, 1, 1]
        N_l[:, 0, 0] = bp[2, :, 0, 0]
        N_l[:, 0, 1] = bp[2, :, 0, 1]
        N_l[:, 1, 0] = bp[2, :, 1, 0]
        N_l[:, 1, 1] = bp[2, :, 1, 1]
    elif field == "TEB":
        C_l_hat = bp[1, :, :, :]
        C_fl = bp[0, :, :, :]
        N_l = bp[2, :, :, :]

    # load the covariance matrix
    M_raw = read_M(exp=experiment, root=root)
    M = np.zeros((9*dim, 9*dim))
    M_inv = np.zeros((9, 9, dim, dim))

    # select the relevant part of the cov matrix
    if field == 'T':
        M[:, :] = M_raw[0::6, 0::6]
    elif field == 'E':
        M[:, :] = M_raw[1::6, 1::6]
    elif field == 'B':
        M[:, :] = M_raw[2::6, 2::6]
    elif field == 'EB':
        M[0::3, 0::3] = M_raw[1::6, 1::6]
        M[1::3, 1::3] = M_raw[2::6, 2::6]
        M[2::3, 2::3] = M_raw[4::6, 4::6]
        M[0::3, 1::3] = M_raw[1::6, 2::6]
        M[1::3, 0::3] = M_raw[2::6, 1::6]
        M[0::3, 2::3] = M_raw[1::6, 4::6]
        M[2::3, 0::3] = M_raw[4::6, 1::6]
        M[1::3, 2::3] = M_raw[2::6, 4::6]
        M[2::3, 1::3] = M_raw[4::6, 2::6]
    elif field == 'TE':
        M[0::3, 0::3] = M_raw[0::6, 0::6]
        M[1::3, 1::3] = M_raw[1::6, 1::6]
        M[2::3, 2::3] = M_raw[3::6, 3::6]
        M[0::3, 1::3] = M_raw[0::6, 1::6]
        M[1::3, 0::3] = M_raw[1::6, 0::6]
        M[0::3, 2::3] = M_raw[0::6, 3::6]
        M[2::3, 0::3] = M_raw[3::6, 0::6]
        M[1::3, 2::3] = M_raw[1::6, 3::6]
        M[2::3, 1::3] = M_raw[3::6, 1::6]
    elif field == 'TB':
        M[0::3, 0::3] = M_raw[0::6, 0::6]
        M[1::3, 1::3] = M_raw[2::6, 2::6]
        M[2::3, 2::3] = M_raw[5::6, 5::6]
        M[0::3, 1::3] = M_raw[0::6, 2::6]
        M[1::3, 0::3] = M_raw[2::6, 0::6]
        M[0::3, 2::3] = M_raw[0::6, 5::6]
        M[2::3, 0::3] = M_raw[5::6, 0::6]
        M[1::3, 2::3] = M_raw[2::6, 5::6]
        M[2::3, 1::3] = M_raw[5::6, 2::6]
    elif field == 'TEB':
        M = M_raw

    # Evaluate inverse of covariance matrix
    M_invp = LA.inv(M)

    # re-organize elements
    for ell in xrange(9):
        for ellp in xrange(9):
            M_inv[ell,ellp,:,:] = M_invp[ell*dim:(ell+1)*dim,ellp*dim:(ellp+1)*dim]

    return C_l, C_l_hat, N_l, C_fl, M_inv, bpwf_l, bpwf_Cs_l
