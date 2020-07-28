import numpy as np
import math
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
# import the python package of the BICEP2 collaboration
import sys 
sys.path.append('.')
from . import bicep_util as bu


class bicep2(Likelihood):

    def __init__(self, path, data, command_line):

        # Define a default value, empty, for which field to use with the
        # likelihood. The proper value should be initialised in the .data file.
        Likelihood.__init__(self, path, data, command_line)

        # Create arrays that will store the values for all the possibly tested
        # fields.
        self.C_l = []
        self.C_l_hat = []
        self.N_l = []
        self.C_fl = []
        self.M_inv = []
        self.bpwf_l = []
        self.bpwf_Cs_l = []

        # Recover all the relevant quantities for the likelihood computation
        # from the BICEP collaboration, which includes the band power window
        # functions (bpwf). It assumes that in the "root" directory, there is a
        # "windows" directory which contains the band power window functions
        # from BICEP2.
        for field in self.fields:
            C_l, C_l_hat, N_l, C_fl, M_inv, bpwf_l, bpwf_Cs_l = bu.init(
                "bicep2",
                field,
                self.data_directory)
            self.C_l.append(C_l)
            self.C_l_hat.append(C_l_hat)
            self.N_l.append(N_l)
            self.C_fl.append(C_fl)
            self.M_inv.append(M_inv)
            self.bpwf_l.append(bpwf_l)
            self.bpwf_Cs_l.append(bpwf_Cs_l)

        # Read the desired max ell from the band power window function.
        self.l_max = max([elem[-1] for elem in self.bpwf_l])

        # Require tensor modes from Class
        arguments = {
            'output': 'tCl pCl lCl',
            'lensing': 'yes',
            'modes': 's, t',
            'l_max_scalars': self.l_max,
            'l_max_tensors': self.l_max,}
        self.need_cosmo_arguments(data, arguments)

    def loglkl(self, cosmo, data):

        # Recover Cl_s from CLASS, which is a dictionary, with the method
        # get_cl from the Likelihood class, because it already makes the
        # conversion to uK^2.
        dict_Cls = self.get_cl(cosmo, self.l_max)

        # Convert the dict to the same format expected by BICEP
        # that is:
        # 0: TT
        # 1: TE
        # 2: EE
        # 3: BB
        # 6: ET, and the rest to 0 (must be an array of width 9)
        # Warnings: BICEP2 expects l*(l+1)*Cl/(2*pi) in units of uK^2
        cosmo_Cls = np.zeros((int(self.l_max+1), 9))
        ell = np.arange(self.l_max+1)

        cosmo_Cls[:, 0] = ell*(ell+1)*dict_Cls['tt']/(2*math.pi)
        cosmo_Cls[:, 1] = ell*(ell+1)*dict_Cls['te']/(2*math.pi)
        cosmo_Cls[:, 2] = ell*(ell+1)*dict_Cls['ee']/(2*math.pi)
        cosmo_Cls[:, 3] = ell*(ell+1)*dict_Cls['bb']/(2*math.pi)
        cosmo_Cls[:, 6] = ell*(ell+1)*dict_Cls['te']/(2*math.pi)

        # Now loop over all the fields
        loglkl = 0
        for index, field in enumerate(self.fields):
            # Get the expectation value of the data considering this theoretical
            # model, for all the required fields.
            expectation_value = bu.calc_expvals(
                ell, cosmo_Cls,
                self.bpwf_l[index], self.bpwf_Cs_l[index])

            # Fill the C_l matrix
            if field == "T":
                self.C_l[index][:, 0, 0] = expectation_value[:, 0]
            elif field == 'E':
                self.C_l[index][:, 0, 0] = expectation_value[:, 2]
            elif field == 'B':
                self.C_l[index][:, 0, 0] = expectation_value[:, 3]
            elif field == "EB":
                self.C_l[index][:, 0, 0] = expectation_value[:, 2]
                self.C_l[index][:, 0, 1] = expectation_value[:, 5]
                self.C_l[index][:, 1, 0] = expectation_value[:, 5]
                self.C_l[index][:, 1, 1] = expectation_value[:, 3]
            elif field == "TB":
                self.C_l[index][:, 0, 0] = expectation_value[:, 0]
                self.C_l[index][:, 0, 1] = expectation_value[:, 4]
                self.C_l[index][:, 1, 0] = expectation_value[:, 5]
                self.C_l[index][:, 1, 1] = expectation_value[:, 3]
            elif field == "TE":
                self.C_l[index][:, 0, 0] = expectation_value[:, 0]
                self.C_l[index][:, 0, 1] = expectation_value[:, 1]
                self.C_l[index][:, 1, 0] = expectation_value[:, 1]
                self.C_l[index][:, 1, 1] = expectation_value[:, 2]
            else:
                raise io_mp.LikelihoodError(
                    "BICEP2 requires a field to compute the likelihood"
                    " which should so far be T, E or B, but was read to"
                    " be '%s'" % field)

            # Add the noise
            self.C_l[index] += self.N_l[index]

            # Actually compute the likelihood
            loglkl += bu.evaluateLikelihood(
                self.C_l[index], self.C_l_hat[index],
                self.C_fl[index], self.M_inv[index])

        return loglkl
