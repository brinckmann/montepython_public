# Lollipop is a low-l polarized likelihood for Planck PR4 data (BB spectrum)

Lollipop is a Planck low-l polarization likelihood based on
cross-power-spectra for which the bias is zero when the noise is
uncorrelated between maps. It uses the approximation presented in
Hamimeche & Lewis (2008), modified as described in Mangilli et
al. (2015) to apply to cross-power spectra. This version is based on
the Planck PR4 data. Cross-spectra are computed on the CMB maps from
Commander component separation applied on each detset-split Planck
frequency maps.

Reference: 
[Tristram et al., A&A, 2021](https://arxiv.org/abs/2010.01139)
[Tristram et al., A&A, 2022](https://arxiv.org/abs/2112.07961)

The code is available here:
[https://github.com/planck-npipe/lollipop](https://github.com/planck-npipe/lollipop)
This is a wrapper for MontePython.

You need to install the code before:
```
pip install planck-2020-lollipop
```

Then get the data, untar and set the variable $COBAYA_PACKAGES_PATH to the local directory:
```
wget https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods/planck_2020_lollipop.tar.gz
tar -zxvf planck_2020_lollipop.tar.gz --directory /path/to/data
export COBAYA_PACKAGES_PATH=/path/to/data
```
