##########################################################
# galaxy-to-matter bias in models with massive neutrinos #
##########################################################

# Galaxies or halos trace only the clustering components (baryon + CDM), which is
# not the same as total matter in presence massive neutrinos (see e.g. 1505.07148)

# Then the galaxy-to-matter bias can be estimated from the power spectra
# P_cb and P_m inferred the Boltzmann solver (see e.g. 1807.04672)

# Written by Sefa Pamuk (2023)

import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

"""
This is not a boost in the traditional sense. It rather is just the ratio of the P_cb and P_mm that shows up when looking at the neutrino induced scale dependent galaxy bias.
"""
def get_boost_neutrino_bias(cosmo, data, lkl, Pk_m_nl_grid, k, z):
    # To obtain the non-linear P_cb spectrum we use the linear-only neutrino approximation
    
    # Get fractions to weigh neutrino and CDM + baryon contributions
    f_cdm = cosmo.Omega0_cdm()/cosmo.Omega_m()
    f_b = cosmo.Omega_b()/cosmo.Omega_m()
    f_cb = f_cdm+f_b
    f_nu = 1-f_cb

    # Obtain linear neutrino auto and neutrino cross CDM+baryon power spectra
    tk, k_grid, z_grid = cosmo.get_transfer_and_k_and_z()
    T_cb = (f_b*tk['d_b']+f_cdm*tk['d_cdm'])/f_cb
    T_nu = tk['d_ncdm[0]']
    pm = cosmo.get_primordial()
    pk_prim = UnivariateSpline(pm['k [1/Mpc]'],pm['P_scalar(k)'])(k_grid)*(2.*np.pi**2)/np.power(k_grid,3)
    Pk_cnu_grid  = T_nu * T_cb * pk_prim[:,None]
    Pk_nunu_grid = T_nu * T_nu * pk_prim[:,None]

    # use the linear-only neutrino approximation to obtain the non-linear clustering power spectrum
    Pk_cb_nl_grid= 1./f_cb**2 * (Pk_m_nl_grid-2*Pk_cnu_grid*f_nu*f_cb-Pk_nunu_grid*f_nu*f_nu)

    z_grid = z_grid[::-1]
    Pk_m_nl_grid = np.flip(Pk_m_nl_grid,axis=1)
    Pk_cb_nl_grid = np.flip(Pk_cb_nl_grid,axis=1)

    Pk_ratio = RectBivariateSpline(k_grid, z_grid, Pk_cb_nl_grid /Pk_m_nl_grid)

    boost_neutrino_bias = np.ones_like(k, 'float64')
    for iz, zi in enumerate(z):
        pknn_mask = np.where((k[:,iz]>lkl.kmin_in_inv_Mpc) & (k[:,iz]<lkl.kmax_in_inv_Mpc))
        boost_neutrino_bias[pknn_mask,iz] = np.reshape(Pk_ratio(k[pknn_mask,iz],zi),boost_neutrino_bias[pknn_mask,iz].shape)

    return boost_neutrino_bias
