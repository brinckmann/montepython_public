######################################################################################################
# Interface module between euclid_photometric_alm likelihood and BCemu emulator of baryonic feedback #
######################################################################################################

# BCemu python package available at https://github.com/sambit-giri/BCemu
# Developped by Sambit Giri and Aurel Schneider, see arXiv:2108.08863 (physical model in arXiv:1810.08629)

# Interface with Euclid likelihood witten by J. Schwagereit and Sefa Pamuk (2023)

import BCemu
import numpy as np
from scipy.interpolate import RectBivariateSpline

def get_boost_baryonic_feedback(cosmo, data, lkl, k, z):

    # baryonic feedback modifications are only applied to k>kmin_bfc
    # it is very computationally expensive to call BCemu at every z in self.z, and it is a very smooth function with z,
    # so it is only called at self.BCemu_k_bins points in k and self.BCemu_z_bins points in z and then the result is
    # splined over all z in self.z. For k>kmax_bfc = 12.5 h/Mpc, the maximum k the emulator is trained on, a constant
    # suppression in k is assumed: BFC(k,z) = BFC(12.5 h/Mpc, z).

    if not hasattr(lkl, "bfcemu"):
        lkl.bfcemu = BCemu.BCM_7param(verbose=lkl.verbose_BCemu)

    log10Mc = data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale']
    nu_Mc   = data.mcmc_parameters['nu_Mc']['current']   * data.mcmc_parameters['nu_Mc']['scale']

    bcemu_dict ={
    'log10Mc' : log10Mc,
    'nu_Mc'   : nu_Mc,
    'mu'      : 0.93,
    'nu_mu'   : 0.0,
    'thej'    : 2.6,
    'nu_thej' : 0.0,
    'gamma'   : 2.25,
    'nu_gamma': 0.0,
    'delta'   : 6.4,
    'nu_delta': 0.0,
    'eta'     : 0.15,
    'nu_eta'  : 0.0,
    'deta'    : 0.14,
    'nu_deta' : 0.06
    }

    Ob = cosmo.Omega_b()
    Om = cosmo.Omega_m()

    fb = Ob/Om
    if fb < 0.1 or fb > 0.25:
        if lkl.verbose: print(" /!\ Skipping point because the baryon fraction is out of bounds!")
        return -1e10

    if log10Mc / 3**nu_Mc < 11 or log10Mc / 3**nu_Mc > 15 :
        if lkl.verbose: print(" /!\ Skipping point because BF parameters are out of bounds!")
        return -1e10

    kmin_bfc = 0.035
    kmax_bfc = 12.5
    k_bfc = np.logspace(np.log10(max(kmin_bfc, lkl.k_min_h_by_Mpc)), np.log10(min(kmax_bfc, lkl.k_max_h_by_Mpc)), lkl.BCemu_k_bins)
    # ^ all have units h/Mpc

    z_bfc = np.linspace(lkl.z[0], min(2, lkl.z[-1]), lkl.BCemu_z_bins)
    BFC = np.zeros((lkl.BCemu_k_bins, lkl.BCemu_z_bins))

    for iz, zi in enumerate(z_bfc):
        BFC[:,iz] = lkl.bfcemu.get_boost(zi,bcemu_dict,k_bfc,fb)

    BFC_interpolator = RectBivariateSpline(k_bfc*cosmo.h(), z_bfc, BFC)
    # ^ gets passed k in units 1/Mpc

    boost_m_nl_BCemu = np.ones_like(k, 'float64')
    for iz, zi in enumerate(z):
        pknn_mask = np.where((k[:,iz]>lkl.kmin_in_inv_Mpc) & (k[:,iz]<lkl.kmax_in_inv_Mpc))
        boost_m_nl_BCemu[pknn_mask, iz] = np.reshape(BFC_interpolator(np.minimum(k[pknn_mask,iz],12.5*cosmo.h()),min(zi, 2))[:,0], boost_m_nl_BCemu[pknn_mask, iz].shape)

    return boost_m_nl_BCemu
