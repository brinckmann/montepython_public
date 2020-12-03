"""
.. module:: prior
    :synopsis: Define the Prior class

.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
"""
from __future__ import print_function
import random as rd
from copy import deepcopy
import io_mp

class Prior(object):
    """
    Store the type of prior associated to a parameter

    """

    def __init__(self, array):
        """
        It takes as an optional input argument the array of the input
        :data:`parameters` defined in the parameter file.

        The current implemented types are 'flat' (default), and 'gaussian',
        which expect also a mean and sigma. Possible extension would take a
        'external', needing to read an external file to read for the
        definition.

        The entry 'prior' of the dictionary :data:`mcmc_parameters` will hold
        an instance of this class. It defines one main function, called
        :func:`draw_from_prior`, that returns a number within the prior volume.

        """

        rd.seed()

        # Test the length of the array, and initialize the type.
        if len(array) == 6:
            # Default behaviour, flat prior
            self.prior_type = 'flat'
        else:
            self.prior_type = array[6].lower()
            # in case of a gaussian prior, one expects two more entries, mu and
            # sigma
            if self.prior_type == 'gaussian':
                try:
                    self.mu = array[7]
                    self.sigma = array[8]
                except IndexError:
                    raise io_mp.ConfigurationError(
                        "You asked for a gaussian prior, but provided no " +
                        "mean nor sigma. Please add them in the parameter " +
                        "file.")

        # Store boundaries for convenient access later
        # Put all fields that are -1 to None to avoid confusion later on.
        self.prior_range = [a if not((a is -1) or (a is None)) else None
                            for a in deepcopy(array[1:3])]

    def draw_from_prior(self):
        """
        Draw a random point from the prior range considering the prior type

        Returns
        -------
        value : float
            A random sample inside the prior region

        """

        if self.prior_type == 'flat':
            return rd.uniform(self.prior_range[0], self.prior_range[1])

        elif self.prior_type == 'gaussian':
            within_bounds = False

            while not within_bounds:
                value = rd.gauss(self.mu, self.sigma)
                # Check for boundaries problem
                within_bounds = calue_within_prior_range(value)

            return value

    def value_within_prior_range(self, value):
        """
        Check for a value being in or outside the prior range

        """
        flag = 0
        if (self.prior_range[0] is not None and value < self.prior_range[0]):
            flag += 1
        elif (self.prior_range[1] is not None and value > self.prior_range[1]):
            flag += 1

        if flag > 0:
            return False
        else:
            return True

    def is_bound(self):
        """
        Checks whether the allowed parameter range is finite

        """
        return (self.prior_range[0] is not None and
                self.prior_range[1] is not None)

    def map_from_unit_interval(self, value):
        """
        Linearly maps a value of the interval [0,1] to the parameter range.

        For the sake of speed, assumes the parameter to be bound to a finite range, \
        which should have been previously checked with :func:`is_bound`

        """
        if self.prior_type == 'flat':
            return (self.prior_range[0] +
                value * (self.prior_range[1] - self.prior_range[0]))
        # Gaussian prior adapted from https://github.com/JohannesBuchner/MultiNest/blob/master/src/priors.f90
        if self.prior_type == 'gaussian':
            if (value <= 1e-16) or ((1 - value) <= 1e-16):
                gaussian_prior = -1e-32
            else:
                gaussian_prior = self.mu + self.sigma * np.sqrt(2) * erfcinv(2*(1-value))
            return(gaussian_prior)
