from montepython.likelihood_class import Likelihood_mpk, Likelihood
import os

class WiggleZ(Likelihood):
    """
    Likelihood for WiggleZ

    """

    def __init__(self, path, data, command_line):
        """
        The structure differs significantly from other likelihoods, in order to
        follow as simply as possible the data structure.

        Each redshift bin in WiggleZ contains a .dataset file with information
        on this redshift bin. The structure to read this has been encoded in
        the class Likelihood_mpk. It will be used also with the next data
        release of SDSS.

        The whole WiggleZ is then made out of the four Likelihood_mpk:
        WiggleZ_a, b, c and d, which are **defined dynamically** thanks to the
        :func:`type` function in python, inheriting from
        :class:`Likelihood_mpk`.

        Some additional keyword arguments are sent to the initialization of
        these classes, in order to use the function
        :meth:`add_common_knowledge`. It then gives a dictionary of shared
        attributes that should be distributed to all four redshift bins.

        """

        Likelihood.__init__(self, path, data, command_line)

        # (JR) initialise each element explicitly, for loop ini with exec()
        # does not work with python3
        WiggleZ_a = type('WiggleZ_a', (Likelihood_mpk, ), {})
        WiggleZ_b = type('WiggleZ_b', (Likelihood_mpk, ), {})
        WiggleZ_c = type('WiggleZ_c', (Likelihood_mpk, ), {})
        WiggleZ_d = type('WiggleZ_d', (Likelihood_mpk, ), {})

        # Initialize one after the other the four independent redshift bins (note:
        # the order in the array self.redshift_bins_files) must be respected !
        self.wigglez_a = WiggleZ_a(
            os.path.join(self.data_directory, self.redshift_bins_files[0]),
            data, command_line, common=True, common_dict=self.dictionary)

        self.wigglez_b = WiggleZ_b(
            os.path.join(self.data_directory, self.redshift_bins_files[1]),
            data, command_line, common=True, common_dict=self.dictionary)

        self.wigglez_c = WiggleZ_c(
            os.path.join(self.data_directory, self.redshift_bins_files[2]),
            data, command_line, common=True, common_dict=self.dictionary)

        self.wigglez_d = WiggleZ_d(
            os.path.join(self.data_directory, self.redshift_bins_files[3]),
            data, command_line, common=True, common_dict=self.dictionary)

    def loglkl(self, cosmo, data):
        # Simply add all the sublikelihoods
        loglkl = 0
        loglkl += self.wigglez_a.loglkl(cosmo, data)
        loglkl += self.wigglez_b.loglkl(cosmo, data)
        loglkl += self.wigglez_c.loglkl(cosmo, data)
        loglkl += self.wigglez_d.loglkl(cosmo, data)
        return loglkl
