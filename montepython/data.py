"""
.. module:: data
   :synopsis: Define the Data and Parameter classes

.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
"""
from __future__ import print_function
import os
import sys
import math
import random as rd
import warnings
import subprocess as sp
import re

import io_mp  # Needs to talk to io_mp.py file for the logging
                               # of parameters
from io_mp import dictitems,dictvalues,dictkeys
import prior
from scipy.optimize import fsolve
import numpy as np

# A modified version of Python dictionary in order to keep track of the order
# in it (much the same as in an array). In case an older version of Python is
# used, this module does not belong to collections. Please remind to put that
# into your PYTHONPATH variable.
try:
    from collections import OrderedDict as od
except ImportError:
    try:
        from ordereddict import OrderedDict as od
    except ImportError:
        raise io_mp.MissingLibraryError(
            "If you are running with Python v2.5 or 2.6, you need" +
            "to manually install the ordereddict package by placing" +
            "the file ordereddict.py in your Python Path")


class Data(object):
    """
    Store all relevant data to communicate between the different modules.

    """

    def __init__(self, command_line, path):
        """
        The Data class holds the cosmological information, the parameters from
        the MCMC run, the information coming from the likelihoods. It is a wide
        collections of information, with in particular two main dictionaries:
        cosmo_arguments and mcmc_parameters.

        It defines several useful **methods**. The following ones are called
        just once, at initialization:

        * :func:`fill_mcmc_parameters`
        * :func:`read_file`
        * :func:`read_version`
        * :func:`group_parameters_in_blocks`

        On the other hand, these two following functions are called every step.

        * :func:`check_for_slow_step`
        * :func:`update_cosmo_arguments`

        Finally, the convenient method :func:`get_mcmc_parameters` will be
        called in many places, to return the proper list of desired parameters.

        It has a number of different **attributes**, and the more important
        ones are listed here:

        * :attr:`boundary_loglike`
        * :attr:`cosmo_arguments`
        * :attr:`mcmc_parameters`
        * :attr:`need_cosmo_update`
        * :attr:`log_flag`

        .. note::

            The `experiments` attribute is extracted from the parameter file,
            and contains the list of likelihoods to use

        .. note::

            The path argument will be used in case it is a first run, and hence
            a new folder is created. If starting from an existing folder, this
            dictionary will be compared with the one extracted from the
            log.param, and will use the latter while warning the user.

        .. warning::

            New in version 2.0.0, you can now specify an oversampling of the
            nuisance parameters, to hasten the execution of a run with
            likelihoods that have many of them. You should specify a new field
            in the parameter file, `data.over_sampling = [1, ...]`, that
            contains a 1 on the first element, and then the over sampling of
            the desired likelihoods. This array must have the same size as the
            number of blocks (1 for the cosmo + 1 for each likelihood with
            varying nuisance parameters). You need to call the code with the
            flag `-j jast` for it to be used.

        To create an instance of this class, one must feed the following
        parameters and keyword arguments:

        Parameters
        ----------
        command_line : NameSpace
            NameSpace containing the input from the :mod:`parser_mp`. It
            stores the input parameter file, the jumping methods, the output
            folder, etc...  Most of the information extracted from the
            command_file will be transformed into :class:`Data` attributes,
            whenever it felt meaningful to do so.
        path : dict
            Contains a dictionary of important local paths. It is used here to
            find the cosmological module location.

        """

        # Initialisation of the random seed
        rd.seed()

        # Store the parameter file
        self.param = command_line.param

        # Recover jumping method from command_line
        self.jumping = command_line.jumping
        self.jumping_factor = command_line.jumping_factor

        # Store the rest of the command line
        self.command_line = command_line

        # Initialise the path dictionnary.
        self.path = {}

        self.boundary_loglike = -1e30
        """
        Define the boundary loglike, the value used to defined a loglike
        that is out of bounds. If a point in the parameter space is affected to
        this value, it will be automatically rejected, hence increasing the
        multiplicity of the last accepted point.
        """

        # Creation of the two main dictionnaries:
        self.cosmo_arguments = {}
        """
        Simple dictionary that will serve as a communication interface with the
        cosmological code. It contains all the parameters for the code that
        will not be set to their default values.  It is updated from
        :attr:`mcmc_parameters`.

        :rtype:   dict
        """
        self.mcmc_parameters = od()
        """
        Ordered dictionary of dictionaries, it contains everything needed by
        the :mod:`mcmc` module for the MCMC procedure.  Every parameter name
        will be the key of a dictionary, containing the initial configuration,
        role, status, last accepted point and current point.

        :rtype: ordereddict
        """

        # Arguments for PyMultiNest
        self.NS_param_names = []
        self.NS_arguments = {}
        """
        Dictionary containing the parameters needed by the PyMultiNest sampler.
        It is filled just before the run of the sampler.  Those parameters not
        defined will be set to the default value of PyMultiNest.

        :rtype: dict
        """

        # Arguments for PyPolyChord
        self.PC_param_names = []
        self.PC_arguments = {}
        """
        Dictionary containing the parameters needed by the PyPolyChord sampler.
        It is filled just before the run of the sampler.  Those parameters not
        defined will be set to the default value of PyPolyChord.

        :rtype: dict
        """

        # Initialise the experiments attribute
        self.experiments = []

        # Initialise the oversampling setting
        self.over_sampling = []
        """
        List storing the respective over sampling of the parameters. The first
        entry, applied to the cosmological parameters, will always be 1.
        Setting it to anything else would simply rescale the whole process. If
        not specified otherwise in the parameter file, all other numbers will
        be set to 1 as well.

        :rtype: list
        """

        # Default value for the number of steps
        self.N = 10

        # Create the variable out, and out_name, which will be initialised
        # later by the :mod:`io_mp` module
        self.out = None
        self.out_name = ''

        # If the parameter file is not a log.param, the path will be read
        # before reading the parameter file.
        if self.param.find('log.param') == -1:
            self.path.update(path)

        # Read from the parameter file to fill properly the mcmc_parameters
        # dictionary.
        self.fill_mcmc_parameters()

        # Test if the recovered path agrees with the one extracted from
        # the configuration file.
        if self.path != {}:
            if not 'root' in self.path:
                self.path.update({'root': path['root']})
            if self.path != path:
                warnings.warn(
                    "Your code location in the log.param file is "
                    "in contradiction with your .conf file. "
                    "I will use the one from log.param.")

        # Determine which cosmological code is in use
        if os.path.isfile(self.path['cosmo']+'/main/class.c'):
            self.cosmological_module_name = 'CLASS'
        else:
            self.cosmological_module_name = None

        # check for MPI
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except ImportError:
            # set all chains to master if no MPI
            rank = 0

        # Recover the cosmological code version (and git hash if relevant).
        # To implement a new cosmological code, please add another case to the
        # test below.
        if self.cosmological_module_name == 'CLASS':
            # Official version number
            common_file_path = os.path.join(
                self.path['cosmo'], 'include', 'common.h')
            with open(common_file_path, 'r') as common_file:
                for line in common_file:
                    if line.find('_VERSION_') != -1:
                        self.version = str(line.split()[-1].replace('"', ''))
                        break
            if not command_line.silent and not rank:
                print('with CLASS %s' % self.version)
            # Git version number and branch
            try:
                # This nul_file helps to get read of a potential useless error
                # message
                with open(os.devnull, "w") as nul_file:
                    self.git_version = str(sp.Popen(
                        ["git", "rev-parse", "HEAD"],
                        cwd=self.path['cosmo'],
                        stdout=sp.PIPE,
                        stderr=nul_file).communicate()[0].strip())
                    self.git_branch = str(sp.Popen(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=self.path['cosmo'],
                        stdout=sp.PIPE,
                        stderr=nul_file).communicate()[0].strip())
                    # Possible conversion from bytestring b'xxx' to normal string
                    if self.git_version.startswith("b'"):
                        self.git_version = self.git_version[2:-1]
                    if self.git_branch.startswith("b'"):
                        self.git_branch = self.git_branch[2:-1]
            except (sp.CalledProcessError, OSError):
                # Note, OSError seems to be raised on some systems, instead of
                # sp.CalledProcessError - which seems to be linked to the
                # existence of os.devnull, so now both error are caught.
                warnings.warn(
                    "Running CLASS from a non version-controlled repository")
                self.git_version, self.git_branch = '', ''

            # If using an existing log.param, read in and compare this number
            # to the already stored one
            if self.param.find('log.param') != -1:
                try:
                    version, git_version, git_branch = self.read_version(
                        self.param_file)
                    if version != self.version:
                        warnings.warn(
                            "Your version of CLASS: %s" % self.version +
                            " does not match the one used previously" +
                            " in this folder (%s)." % version +
                            " Proceed with caution")
                    else:
                        if self.git_branch != git_branch:
                            warnings.warn(
                                "CLASS set to branch %s" % self.git_branch +
                                ", wrt. the one used in the log.param:" +
                                " %s." % git_branch)
                        if self.git_version != git_version:
                            warnings.warn(
                                "CLASS set to version %s" % self.git_version +
                                ", wrt. the one used in the log.param:" +
                                " %s." % git_version)

                except AttributeError:
                    # This error is raised when the regular expression match
                    # failed - due to comparing to an old log.param that did
                    # not have this feature properly implemented. Ignore this.
                    pass

        else:
            raise io_mp.CosmologicalModuleError(
                "Could not find any of the registered cosmological modules, such as {CLASS}."
                " The code tried to search for "+self.path['cosmo']+'/main/class.c'+" but the file did not exist."
                " If you want to check for another cosmological module version"
                " please add an elif clause to this part of the code.")

        # End of initialisation with the parameter file
        self.param_file.close()

        self.log_flag = False
        """
        Stores the information whether or not the likelihood data files need to
        be written down in the log.param file. Initially at False.

        :rtype: bool
        """

        self.need_cosmo_update = True
        """
        `added in version 1.1.1`. It stores the truth value of whether the
        cosmological block of parameters was changed from one step to another.
        See :meth:`group_parameters_in_blocks`

        :rtype: bool
        """

        # logging the parameter file (only if folder does not exist !)
        ## temporary variable for readability
        log_param = os.path.join(command_line.folder, 'log.param')

        if (os.path.exists(command_line.folder) and
                not os.path.exists(log_param)):
            if command_line.param is not None:
                warnings.warn(
                    "Detecting empty folder, logging the parameter file")
                io_mp.log_parameters(self, command_line)
                self.log_flag = True
        if not os.path.exists(command_line.folder):
            os.makedirs(command_line.folder)
            # Logging of parameters
            io_mp.log_parameters(self, command_line)
            self.log_flag = True

        if not command_line.silent and not rank:
            sys.stdout.write('\nTesting likelihoods for:\n ->')
            print(', '.join(self.experiments)+'\n')

        self.initialise_likelihoods(self.experiments)

        # Storing parameters by blocks of speed
        self.group_parameters_in_blocks()

        # Finally, log the cosmo_arguments used. This comes in the end, because
        # it can be modified inside the likelihoods init functions
        if self.log_flag:
            io_mp.log_cosmo_arguments(self, command_line)
            io_mp.log_default_configuration(self, command_line)

        # Log plotting parameter names file for compatibility with GetDist
        io_mp.log_parameter_names(self, command_line)

    def fill_mcmc_parameters(self):
        """
        Initializes the ordered dictionary :attr:`mcmc_parameters` from
        the input parameter file.

        It uses :meth:`read_file`, and initializes instances of
        :class:`parameter` to actually fill in :attr:`mcmc_parameters`.

        """

        # Define temporary quantities, only to simplify the input in the
        # parameter file
        self.parameters = od()

        # Read from the parameter file everything
        try:
            self.param_file = open(self.param, 'r')
        except IOError:
            raise io_mp.ConfigurationError(
                "Error in initializing the Data class, the parameter file " +
                "{0} does not point to a proper file".format(self.param))
        # In case the parameter file is a log.param, scan first once the file
        # to extract only the path dictionnary.
        if self.param.find('log.param') != -1:
            self.read_file(self.param, 'data', field='path')
        self.read_file(self.param, 'data')

        # Test here whether the number of parameters extracted correspond to
        # the number of lines (to make sure no doublon is present)
        # Changed to ignore '#' after a data.parameters[x]=y line
        number_of_parameters=0
        for line in open(self.param,'r'):
          if(line):
            if(line.find("#") != -1):
              line = line.split("#")[0]
            if(line.find('data.parameters[') != -1):
              number_of_parameters+=1
        if number_of_parameters != len(self.parameters):
            raise io_mp.ConfigurationError(
                "You probably have two lines in your parameter files with "
                "the same parameter name. This is most probably an error, "
                "which will cause problems down the line. Please fix this.")

        # Do the same for every experiments - but only if you are starting a
        # new folder. Otherwise, this step will actually be done when
        # initializing the likelihood.

        # [NS] the below line is not currently used to avoid a bug with using inputs from the param file
        # if self.param.find('log.param') == -1:
        for experiment in self.experiments:
            self.read_file(self.param, experiment, separate=True)

        # Finally create all the instances of the Parameter given the input.
        for key, value in dictitems(self.parameters):
            self.mcmc_parameters[key] = Parameter(value, key)

            # When there is no prior edge requested, the syntax consists in setting it to 'None' in the input file.
            # There is also an old syntax which is deprecated: '-1'.
            # We still allow for that, but just after parsing it, we substitute it with 'None'.
            # When the user really wants a prior edge in -1, he can write -1.0, then the next lines will not substitute it.
            for i in [1,2]:
                if (str(self.mcmc_parameters[key]['initial'][i]) == '-1'):
                    self.mcmc_parameters[key]['initial'][i] = None

        """
        Transform from parameters dictionary to mcmc_parameters dictionary of
        instances from the class :class:`parameter` (inheriting from dict)
        """

    def initialise_likelihoods(self, experiments):
        """
        Given an array of experiments, return an ordered dict of instances

        .. Note::

            in the __init__ method, experiments is naturally self.experiments,
            but it is useful to keep it as a parameter, for the case of
            importance sampling.

        """

        self.lkl = od()
        # adding the likelihood directory to the path, to import the module
        # then, for each library, calling an instance of the likelihood.
        # Beware, though, if you add new likelihoods, they should go to the
        # folder likelihoods/yourlike/yourlike.py, and contain a yourlike.data,
        # otherwise the following set of commands will not work anymore.

        # For the logging if log_flag is True, each likelihood will log its
        # parameters

        # Due to problems in relative import, this line must be there. Until a
        # better solution is found. It adds the root folder of the MontePython
        # used as the first element in the sys.path
        sys.path.insert(0, self.path['root'])

        for elem in experiments:

            folder = os.path.abspath(os.path.join(
                self.path['MontePython'], "likelihoods", "%s" % elem))
            # add the folder of the likelihood to the path of libraries to...
            # ... import easily the likelihood.py program
            try:
                sys.path.insert(0, os.path.abspath(self.path['MontePython']))
                exec("from likelihoods.%s import %s" % (
                    elem, elem))
            except ImportError as message:
                raise io_mp.ConfigurationError(
                    "Trying to import the %s likelihood" % elem +
                    " as asked in the parameter file, and failed."
                    " Please make sure it is in the `montepython/"
                    "likelihoods` folder, and is a proper python "
                    "module. Check also that the name of the class"
                    " defined in the __init__.py matches the name "
                    "of the folder. In case this is not enough, "
                    "here is the original message: %s\n" % message)
            # Initialize the likelihoods. Depending on the values of
            # command_line and log_flag, the routine will call slightly
            # different things. If log_flag is True, the log.param will be
            # appended.
            try:
                exec("self.lkl['%s'] = %s('%s/%s.data',\
                    self, self.command_line)" % (
                    elem, elem, folder, elem))
            except KeyError as e:
                if e.find('clik') != -1:
                    raise io_mp.ConfigurationError(
                        "You should provide a 'clik' entry in the dictionary "
                        "path defined in the file default.conf")
                else:
                    raise io_mp.ConfigurationError(
                        "The following key: '%s' was not found" % e)

    # Custom function to replace string occurances
    replace_string_type = []
    def replace_string_occurance(self,m):
        if m.group(1) is None:
            return m.group()
        else:
            return m.group().replace(self.replace_string_type[0],self.replace_string_type[1])

    def read_file(self, param, structure, field='', separate=False):
        """
        Execute all lines concerning the Data class from a parameter file

        All lines starting with `data.` will be replaced by `self.`, so the
        current instance of the class will contain all the information.

        .. note::

            A rstrip() was added at the end, because of an incomprehensible bug
            on some systems that imagined some inexistent characters at the end
            of the line... Now should work

        .. note::

            A security should be added to protect from obvious attacks.

        Parameters
        ----------
        param : str
            Name of the parameter file
        structure : str
            Name of the class entries we want to execute (mainly, data, or any
            other likelihood)

        Keyword Arguments
        -----------------
        field : str
            If nothing is specified, this routine will execute all the lines
            corresponding to the `structure` parameters. If you specify a
            specific field, like `path`, only this field will be read and
            executed.
        separate : bool
            If this flag is set to True, a container class will be created for
            the structure field, so instead of appending to the namespace of
            the data instance, it will append to a sub-namespace named in the
            same way that the desired structure. This is used to extract custom
            values from the likelihoods, allowing to specify values for the
            likelihood directly in the parameter file.

        """
        if separate:
            exec("self.%s = Container()" % structure)
        with open(param, 'r') as param_file:
            for line in param_file:
                if line:
                    # Only use contents in param_file before any '#' character
                    if line.find("#") != -1:
                        line = line.split("#")[0]
                    lhs = line.split('=')[0]
                    if lhs.find(structure+'.') != -1:
                        if field:
                            # If field is not an empty string, you want to skip
                            # the execution of the line (exec statement) if you
                            # do not find the exact searched field
                            if lhs.find('.'.join([structure, field])) == -1:
                                continue
                            self.replace_string_type = [structure+'.','self.']
                        elif structure == "data":
                            self.replace_string_type = [structure+'.','self.']
                        else:
                            self.replace_string_type = [structure+'.','self.'+structure+'.']
                        # With this we can read from inside a .param file
                        statement = re.sub(r"'[^']*'|([^']*)", self.replace_string_occurance, line) #Only replace outside of quotation marks
                        exec(statement.rstrip())

    def group_parameters_in_blocks(self):
        """
        Regroup mcmc parameters by blocks of same speed

        This method divides all varying parameters from :attr:`mcmc_parameters`
        into as many categories as there are likelihoods, plus one (the slow
        block of cosmological parameters).

        It creates the attribute :attr:`block_parameters`, to be used in the
        module :mod:`mcmc`.

        .. note::

            It does not compute by any mean the real speed of each parameter,
            instead, every parameter belonging to the same likelihood will
            be considered as fast as its neighbour.

        .. warning::

            It assumes that the nuisance parameters are already written
            sequentially, and grouped together (not necessarily in the order
            described in :attr:`experiments`). If you mix up the different
            nuisance parameters in the .param file, this routine will not
            method as intended. It also assumes that the cosmological
            parameters are written at the beginning of the file.

        """
        array = []
        # First obvious block is all cosmological parameters
        array.append(len(self.get_mcmc_parameters(['varying', 'cosmo'])))
        # Then, store all nuisance parameters
        nuisance = self.get_mcmc_parameters(['varying', 'nuisance'])

        # Create an array to keep track of the already taken into account
        # nuisance parameters. This will come in handy when using likelihoods
        # that share some nuisance parameters.
        used_nuisance = []
        for likelihood in dictvalues(self.lkl):
            count = 0
            for elem in nuisance:
                if elem in likelihood.nuisance:
                    if elem not in used_nuisance:
                        count += 1
                        used_nuisance.append(elem)
            likelihood.varying_nuisance_parameters = count

        # Then circle through them
        index = 0
        while index < len(nuisance):
            elem = nuisance[index]
            flag = False
            # For each one, check if they belong to a likelihood
            for likelihood in dictvalues(self.lkl):
                if (elem in likelihood.nuisance) and (index < len(nuisance)):
                    # If yes, store the number of nuisance parameters needed
                    # for this likelihood.
                    flag = True
                    array.append(
                        likelihood.varying_nuisance_parameters+array[-1])
                    index += likelihood.varying_nuisance_parameters
                    continue
            if not flag:
                # If the loop reaches this part, it means this nuisance
                # parameter was associated with no likelihood: this should not
                # happen
                raise io_mp.ConfigurationError(
                    "nuisance parameter %s " % elem +
                    "is associated to no likelihood")
        # Store the result
        self.block_parameters = array

        # Setting a default value for the over_sampling array
        if not self.over_sampling:
            self.over_sampling = [1 for _ in range(len(self.block_parameters))]
        # Test that the over_sampling list has the same size as
        # block_parameters.
        else:
            try:
                assert len(self.block_parameters) == len(self.over_sampling)
            except AssertionError:
                raise io_mp.ConfigurationError(
                    "The length of the over_sampling field should be"
                    " equal to the number of blocks (one for cosmological "
                    "parameters, plus one for each likelihood with "
                    "nuisance parameters)")

        # Create a list of indices corresponding of the oversampling strategy
        self.assign_over_sampling_indices()

    def assign_over_sampling_indices(self):
        """
        Create the list of varied parameters given the oversampling
        """
        self.over_sampling_indices = []
        for index in range(len(self.get_mcmc_parameters(['varying']))):
            if index == 0:
                self.over_sampling_indices.append(index)
            else:
                block_index = self.block_parameters.index(
                    [i for i in self.block_parameters if index < i][0])
                for _ in range(self.over_sampling[block_index]):
                    self.over_sampling_indices.append(index)

    def read_version(self, param_file):
        """
        Extract version and subversion from an existing log.param
        """
        # Read the first line (cosmological code version)
        first_line = param_file.readline()
        param_file.seek(0)
        regexp = re.match(
            ".*\(branch: (.*), hash: (.*)\).*",
            first_line)
        version = first_line.split()[1]
        git_branch, git_version = regexp.groups()
        # Possible conversion from bytestring b'xxx' to normal string
        if git_branch.startswith("b'"):
          git_branch = git_branch[2:-1]
        if git_version.startswith("b'"):
          git_version = git_version[2:-1]
        return str(version), str(git_version), str(git_branch)

    def get_mcmc_parameters(self, table_of_strings):
        """
        Returns an ordered array of parameter names filtered by
        `table_of_strings`.

        Parameters
        ----------
        table_of_strings : list
            List of strings whose role and status must be matched by a
            parameter. For instance,

            >>> data.get_mcmc_parameters(['varying'])
            ['omega_b', 'h', 'amplitude', 'other']

            will return a list of all the varying parameters, both
            cosmological and nuisance ones (derived parameters being `fixed`,
            they wont be part of this list). Instead,

            >>> data.get_mcmc_parameters(['nuisance', 'varying'])
            ['amplitude', 'other']

            will only return the nuisance parameters that are being varied.

        """
        table = []
        for key, value in dictitems(self.mcmc_parameters):
            number = 0
            for subvalue in dictvalues(value):
                for string in table_of_strings:
                    if subvalue == string:
                        number += 1
            if number == len(table_of_strings):
                table.append(key)
        return table

    def check_for_slow_step(self, new_step):
        """
        Check whether the value of cosmological parameters were
        changed, and if no, skip computation of the cosmology.

        """
        parameter_names = self.get_mcmc_parameters(['varying'])
        cosmo_names = self.get_mcmc_parameters(['cosmo'])

        need_change = 0

        # For all elements in the varying parameters:
        for elem in parameter_names:
            i = parameter_names.index(elem)
            # If it is a cosmological parameter
            if elem in cosmo_names:
                if self.mcmc_parameters[elem]['current'] != new_step[i]:
                    need_change += 1

        # If any cosmological value was changed,
        if need_change > 0:
            self.need_cosmo_update = True
        else:
            self.need_cosmo_update = False

        for likelihood in dictvalues(self.lkl):
            # If the cosmology changed, you need to recompute the likelihood
            # anyway
            if self.need_cosmo_update:
                likelihood.need_update = True
                continue
            # Otherwise, check if the nuisance parameters of this likelihood
            # were changed
            need_change = 0
            for elem in parameter_names:
                i = parameter_names.index(elem)
                if elem in likelihood.nuisance:
                    if self.mcmc_parameters[elem]['current'] != new_step[i]:
                        need_change += 1
            if need_change > 0:
                likelihood.need_update = True
            else:
                likelihood.need_update = False

    def update_cosmo_arguments(self):
        """
        Put in :attr:`cosmo_arguments` the current values of
        :attr:`mcmc_parameters`

        This method is called at every step in the Markov chain, to update the
        dictionary. In the Markov chain, the scale is not remembered, so one
        has to apply it before giving it to the cosmological code.

        .. note::

            When you want to define new parameters in the Markov chain that do
            not have a one to one correspondance to a cosmological name, you
            can redefine its behaviour here. You will find in the source
            several such examples.

        .. note::

            For complex CLASS parameters, that expect a string of numbers
            separated with commas, you can now use the name of the argument,
            for instance :code:`m_ncdm`, then append a double underscore and a
            number. So if you run with two cosmological parameters,
            :code:`m_ncdm__1` and :code:`m_ncdm__2`, this function will
            automatically concatenate the two and feed class :code:`m_ncdm`.
            You still have to make sure that the other variables are properly
            set, like :code:`N_ncdm` to 2, in this example.

        """
        # For all elements in any cosmological parameters
        for elem in self.get_mcmc_parameters(['cosmo']):
            # Fill in the dictionnary with the current value of parameters
            self.cosmo_arguments[elem] = \
                self.mcmc_parameters[elem]['current'] *\
                self.mcmc_parameters[elem]['scale']

        # For all elements in the cosmological arguments list iterate through
        # to see if any need to be updated. Normally not necessary, but this
        # allowes us to catch and edit parameters that aren't being varied
        # as cosmological parameters (which is done in the following part).
        # Be careful as this list includes the cosmological parameters as well.
        for elem in self.cosmo_arguments:
            if elem == 'sBBN file':
                # Due to a change in CLASS v2.10.0 we may need to change the BBN
                # file path for backwards compatibility. By default this will
                # now pick the BBN path based on the CLASS version number. Use
                # the flag data.custom_bbn_file = True to override this behavior.
                try:
                    self.custom_bbn_file
                except:
                    self.custom_bbn_file = False
                if not self.custom_bbn_file and self.cosmological_module_name == 'CLASS':
                    if ('2.10' in self.version) or float(self.version[1])>2:
                        if 'external' in self.cosmo_arguments['sBBN file']:
                            continue
                        else:
                            self.cosmo_arguments['sBBN file'] = self.path['cosmo']+'/external/bbn/sBBN.dat'
                    else:
                        if 'external' in self.cosmo_arguments['sBBN file']:
                            self.cosmo_arguments['sBBN file'] = self.path['cosmo']+'/bbn/sBBN.dat'
                        else:
                            continue
                    print(self.version)

        # For all elements in the cosmological parameters from the mcmc list,
        # translate any-one that is not directly a CLASS parameter into one.
        # The try: except: syntax ensures that the first call
        for elem in self.get_mcmc_parameters(['cosmo']):
            # infer h from Omega_Lambda and delete Omega_Lambda
            if elem == 'Omega_Lambda':
                omega_b = self.cosmo_arguments['omega_b']
                omega_cdm = self.cosmo_arguments['omega_cdm']
                Omega_Lambda = self.cosmo_arguments['Omega_Lambda']
                self.cosmo_arguments['h'] = math.sqrt(
                    (omega_b+omega_cdm) / (1.-Omega_Lambda))
                del self.cosmo_arguments[elem]
            # infer omega_cdm from Omega_L and delete Omega_L
            elif elem == 'Omega_L':
                omega_b = self.cosmo_arguments['omega_b']
                h = self.cosmo_arguments['h']
                Omega_L = self.cosmo_arguments['Omega_L']
                self.cosmo_arguments['omega_cdm'] = (1.-Omega_L)*h*h-omega_b
                del self.cosmo_arguments[elem]
            elif elem == 'ln10^{10}A_s':
                self.cosmo_arguments['A_s'] = math.exp(
                    self.cosmo_arguments[elem]) / 1.e10
                del self.cosmo_arguments[elem]
            elif elem == 'exp_m_2_tau_As':
                tau_reio = self.cosmo_arguments['tau_reio']
                self.cosmo_arguments['A_s'] = self.cosmo_arguments[elem] * \
                    math.exp(2.*tau_reio)
                del self.cosmo_arguments[elem]
            elif elem == 'f_cdi':
                self.cosmo_arguments['n_cdi'] = self.cosmo_arguments['n_s']
            elif elem == 'beta':
                self.cosmo_arguments['alpha'] = 2.*self.cosmo_arguments['beta']
            elif elem == 'M_tot_NH' or elem == 'M_nu_NH' or elem == 'Mnu_NH' or elem == '{\sum}m_nu_NH' or elem == '{\sum}mnu_NH':
                # By T. Brinckmann
                # Normal hierarchy massive neutrinos. Calculates the individual
                # neutrino masses from M_tot_NH and deletes M_tot_NH
                if not self.cosmo_arguments['N_ncdm'] == 3:
                    raise ValueError(
                        "N_ncdm is not equal to 3."
                        " This value should be exactly 3.")
                # From Esteban et al. 2016: https://arxiv.org/abs/1611.01514
                delta_m_squared_atm=2.524e-3 #2.45e-3
                delta_m_squared_sol=7.50e-5 #7.50e-5
                #m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
                m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot - m1 - (d_m_sq_sol + m1**2.)**0.5 - (d_m_sq_atm + m1**2.)**0.5
                m1,opt_output,success,output_message = fsolve(m1_func,self.cosmo_arguments[elem]/3.,(self.cosmo_arguments[elem],delta_m_squared_atm,delta_m_squared_sol),full_output=True)
                if not success == 1:
                    raise ValueError(
                        "Failed to estimate m1. Reason: "+output_message+
                        " Exiting run.")
                m1 = m1[0]
                m2 = (delta_m_squared_sol + m1**2.)**0.5
                #m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
                m3 = (delta_m_squared_atm + m1**2.)**0.5
                if m1+m2+m3 > self.cosmo_arguments[elem]+0.001*self.cosmo_arguments[elem]:
                    raise ValueError(
                        "Failed to estimate m1 resulting in sum(m_i) > M_tot."
                        " Exiting run.")
                self.cosmo_arguments['m_ncdm'] = r'%g, %g, %g' % (m1,m2,m3)
                del self.cosmo_arguments[elem]
            elif elem == 'M_tot_IH' or elem == 'M_nu_IH' or elem == 'Mnu_IH' or elem == '{\sum}m_nu_IH' or elem == '{\sum}mnu_IH':
                # By T. Brinckmann
                # Inverted hierarchy massive neutrinos. Calculates the individual
                # neutrino masses from M_tot_IH and deletes M_tot_IH
                if not self.cosmo_arguments['N_ncdm'] == 3:
                    raise ValueError(
                        "N_ncdm is not equal to 3."
                        " This value should be exactly 3.")
                # From Esteban et al. 2016: https://arxiv.org/abs/1611.01514
                delta_m_squared_atm=-2.514e-3 #-2.45e-3
                delta_m_squared_sol=7.50e-5 #7.50e-5
                #m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
                m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot - m1 - (d_m_sq_sol + m1**2.)**0.5 - (abs(d_m_sq_atm + d_m_sq_sol + m1**2.))**0.5
                m1,opt_output,success,output_message = fsolve(m1_func,self.cosmo_arguments[elem]/2.,(self.cosmo_arguments[elem],delta_m_squared_atm,delta_m_squared_sol),full_output=True)
                if not success == 1:
                    raise ValueError(
                        "Failed to estimate m1. Reason: "+output_message+
                        " Exiting run.")
                m1 = m1[0]
                m2 = (delta_m_squared_sol + m1**2.)**0.5
                #m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
                m3 = (delta_m_squared_atm + m2**2.)**0.5
                if m1+m2+m3 > self.cosmo_arguments[elem]+0.001*self.cosmo_arguments[elem]:
                    raise ValueError(
                        "Failed to estimate m1 resulting in sum(m_i) > M_tot."
                        "Exiting run.")
                if delta_m_squared_atm + delta_m_squared_sol + m1**2. < 0.:
                    raise ValueError(
                        "Failed to correctly estimate m1. Found m1^2 = %f < %f,"
                        "but m1^2 should always be greater than this value." % (m1**2.,- delta_m_squared_sol - delta_m_squared_atm))
                self.cosmo_arguments['m_ncdm'] = r'%g, %g, %g' % (m1,m2,m3)
                del self.cosmo_arguments[elem]
            elif elem == 'M_tot' or elem == 'M_nu' or elem == 'Mnu' or elem == '{\sum}m_nu' or elem == '{\sum}mnu':
                # By T. Brinckmann
                # Massive neutrinos with identical non-zero mass. Calculates the
                # individual neutrino masses from M_tot and deletes M_tot
                if not self.cosmo_arguments['N_ncdm'] == 1:
                    raise ValueError(
                        "N_ncdm is not equal to 1."
                        " This value should be exactly 1.")
                self.cosmo_arguments['m_ncdm'] = self.cosmo_arguments[elem]/self.cosmo_arguments['deg_ncdm']
                del self.cosmo_arguments[elem]
            elif elem == 'm_s_eff':
                # By T. Brinckmann
                # conversion from effective sterile neutrino mass to physical sterile neutrino mass, assuming that this
                # is the ncdm species number 2 and that it is Dodelson-Widrow like (i.e same temperature as active neutrinos)
                #print self.cosmo_arguments
                #self.cosmo_arguments['m_ncdm__2'] = self.cosmo_arguments['deg_ncdm__2']*self.cosmo_arguments[elem]
                m_s_eff = self.cosmo_arguments[elem]/self.cosmo_arguments['deg_ncdm__2']
                self.cosmo_arguments['m_ncdm'] = r'%g, %g' % (float(self.cosmo_arguments['m_ncdm']), m_s_eff)
                del self.cosmo_arguments[elem]
            # infer omega_cdm from omega_m, omega_b and omega_nu (omega_nu = sum(m_nu) / 93.14)
            # [NS] moved after the determination of self.cosmo_arguments['m_ncdm']
            # TB: generalized to arbitrary number of massive species and more parameter names
            # applied MCMC parameter scaling only for small omega parameters, betting no one rescales the rest
            elif elem == 'omega_m':
                # MCMC parameter scaling for omega_m, if relevant
                scale = 1.
                if 'omega_m' in self.mcmc_parameters:
                    scale = self.mcmc_parameters['omega_m']['scale']
                omega_m = self.cosmo_arguments['omega_m'] * scale

                # Check for massive neutrinos and compute omega_nu if relevant
                if 'N_ncdm' in self.cosmo_arguments and self.cosmo_arguments['N_ncdm']>0:
                    if 'omega_ncdm' in self.cosmo_arguments:
                        # MCMC parameter scaling for omega_nu, if relevant
                        scale = 1.
                        if 'omega_ncdm' in self.mcmc_parameters:
                            scale = self.mcmc_parameters['omega_ncdm']['scale']
                        omega_nu = self.cosmo_arguments['omega_ncdm'] * scale
                    elif 'Omega_ncdm' in self.cosmo_arguments:
                        # We need h to go from Omega_nu to omega_nu, return error if not provided
                        if 'h' in self.cosmo_arguments:
                            omega_nu = self.cosmo_arguments['Omega_ncdm']*self.cosmo_arguments['h']**2
                        elif 'H0' in self.cosmo_arguments:
                            omega_nu = self.cosmo_arguments['Omega_ncdm']*(self.cosmo_arguments['H0']/100.)**2
                        else:
                            raise io_mp.ConfigurationError("Provided Omega_ncdm, but could not find h directy,",
                                                           "since neither of {h,H0} are defined")
                    elif 'M_tot' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['M_tot'] / 93.14
                    elif 'M_nu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['M_nu'] / 93.14
                    elif 'Mnu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['Mnu'] / 93.14
                    elif '{\sum}m_nu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['{\sum}m_nu'] / 93.14
                    elif '{\sum}mnu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['{\sum}mnu'] / 93.14
                    elif 'm_ncdm' in self.cosmo_arguments:
                        # Check number of massive neutrinos, assume 1 (CLASS default) if not specified
                        if 'deg_ncdm' in self.cosmo_arguments:
                            if type(self.cosmo_arguments['deg_ncdm']) == str:
                                deg_ncdm = np.array(self.cosmo_arguments['deg_ncdm'].split(','), dtype='float')
                            else:
                                deg_ncdm = np.array(float(self.cosmo_arguments['deg_ncdm']))
                        else:
                            deg_ncdm = np.array(1.)

                        # Compute neutrino mass sum based on number of massive neutrinos
                        if type(self.cosmo_arguments['m_ncdm']) == str:
                            m_ncdm = np.array(self.cosmo_arguments['m_ncdm'].split(','), dtype='float')
                        else:
                            m_ncdm = np.array(float(self.cosmo_arguments['m_ncdm']))
                        omega_nu = np.dot(deg_ncdm,m_ncdm) / 93.14
                    else:
                        raise io_mp.ConfigurationError("N_ncdm is greater than 0, but couldn't identify any of",
                                                       "{m_ncdm,omega_ncdm,Omega_ncdm,M_tot,M_nu,Mnu,{\sum}m_nu,{\sum}mnu}")
                else:
                    omega_nu = 0.

                # Check for omega_b or Omega_b, return error if not provided
                if 'omega_b' in self.cosmo_arguments:
                    # MCMC parameter scaling for omega_b, if relevant
                    scale = 1.
                    if 'omega_b' in self.mcmc_parameters:
                        scale = self.mcmc_parameters['omega_b']['scale']
                    omega_b = self.cosmo_arguments['omega_b'] * scale
                elif 'Omega_b' in self.cosmo_arguments:
                    # We need h to go from Omega_b to omega_b, return error if not provided
                    if 'h' in self.cosmo_arguments:
                        omega_b = self.cosmo_arguments['Omega_b']*self.cosmo_arguments['h']**2
                    elif 'H0' in self.cosmo_arguments:
                        omega_b = self.cosmo_arguments['Omega_b']*(self.cosmo_arguments['H0']/100.)**2
                    else:
                        raise io_mp.ConfigurationError("Could not find h directy, since neither of {h,H0} are defined")
                else:
                    raise io_mp.ConfigurationError("Could not indentify any of {omega_b,Omega_b} for the definition of omega_m")

                # If you have additional massive dark species they should be included below
                del self.cosmo_arguments[elem]
            # Same as for 'omega_m', just with 'Omega_m' instead
            elif elem == 'Omega_m':
                Omega_m = self.cosmo_arguments['Omega_m']

                # Check for massive neutrinos and compute omega_nu if relevant
                if 'N_ncdm' in self.cosmo_arguments and self.cosmo_arguments['N_ncdm']>0:
                    if 'Omega_ncdm' in self.cosmo_arguments:
                        Omega_nu = self.cosmo_arguments['Omega_ncdm']
                    elif 'omega_ncdm' in self.cosmo_arguments:
                        # MCMC parameter scaling for omega_ncdm, if relevant
                        scale = 1.
                        if 'omega_ncdm' in self.mcmc_parameters:
                            scale = self.mcmc_parameters['omega_ncdm']['scale']
                        omega_nu = self.cosmo_arguments['omega_ncdm'] * scale
                    elif 'M_tot' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['M_tot'] / 93.14
                    elif 'M_nu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['M_nu'] / 93.14
                    elif 'Mnu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['Mnu'] / 93.14
                    elif '{\sum}m_nu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['{\sum}m_nu'] / 93.14
                    elif '{\sum}mnu' in self.cosmo_arguments:
                        omega_nu = self.cosmo_arguments['{\sum}mnu'] / 93.14
                    elif 'm_ncdm' in self.cosmo_arguments:
                        # Check number of massive neutrinos, assume 1 (CLASS default) if not specified
                        if 'deg_ncdm' in self.cosmo_arguments:
                            if type(self.cosmo_arguments['deg_ncdm']) == str:
                                deg_ncdm = np.array(self.cosmo_arguments['deg_ncdm'].split(','), dtype='float')
                            else:
                                deg_ncdm = np.array(float(self.cosmo_arguments['deg_ncdm']))
                        else:
                            deg_ncdm = np.array(1.)

                        # Compute neutrino mass sum based on number of massive neutrinos
                        if type(self.cosmo_arguments['m_ncdm']) == str:
                            m_ncdm = np.array(self.cosmo_arguments['m_ncdm'].split(','), dtype='float')
                        else:
                            m_ncdm = np.array(float(self.cosmo_arguments['m_ncdm']))
                        omega_nu = np.dot(deg_ncdm,m_ncdm) / 93.14
                    else:
                        raise io_mp.ConfigurationError("N_ncdm is greater than 0, but couldn't identify any of",
                                                       "{m_ncdm,omega_ncdm,Omega_ncdm,M_tot,M_nu,Mnu,{\sum}m_nu,{\sum}mnu}")
                    # We need h to go from omega_nu to Omega_nu if the former was computed, return error if not provided
                    try:
                        if omega_nu:
                            if 'h' in self.cosmo_arguments:
                                Omega_nu = omega_nu / self.cosmo_arguments['h']**2
                            elif 'H0' in self.cosmo_arguments:
                                Omega_nu = omega_nu / (self.cosmo_arguments['H0']/100.)**2
                            else:
                                raise io_mp.ConfigurationError("Provided Omega_ncdm, but could not find h directy,",
                                                               "since neither of {h,H0} are defined")
                    except NameError:
                        pass
                else:
                    Omega_nu = 0.

                # Check for omega_b or Omega_b, return error if not provided
                if 'Omega_b' in self.cosmo_arguments:
                    Omega_b = self.cosmo_arguments['Omega_b']
                elif 'omega_b' in self.cosmo_arguments:
                    # MCMC parameter scaling for omega_b, if relevant
                    scale = 1.
                    if 'omega_b' in self.mcmc_parameters:
                        scale = self.mcmc_parameters['omega_b']['scale']
                    # We need h to go from omega_b to Omega_b, return error if not provided
                    if "h" in self.cosmo_arguments:
                        Omega_b = (self.cosmo_arguments['omega_b'] * scale)/self.cosmo_arguments['h']**2
                    elif "H0" in self.cosmo_arguments:
                        Omega_b = (self.cosmo_arguments['omega_b'] * scale)/(self.cosmo_arguments['H0']/100.)**2
                    else:
                        raise io_mp.ConfigurationError("Could not find h directy, since neither of {h,H0} are defined")
                else:
                    raise io_mp.ConfigurationError("Could not indentify any of {omega_b,Omega_b} for the definition of Omega_m")

                # If you have additional massive dark species they should be included below
                self.cosmo_arguments['Omega_cdm'] = Omega_m - Omega_b - Omega_nu
                del self.cosmo_arguments[elem]
            elif elem == 'log10N_dg':
                self.cosmo_arguments['N_dg'] = 10**(self.cosmo_arguments[elem])
                del self.cosmo_arguments[elem]
            elif elem == 'log10fn':
                self.cosmo_arguments['f_nadm'] = 10**(self.cosmo_arguments[elem])
                del self.cosmo_arguments[elem]
            elif elem == 'log10Gamma':
                self.cosmo_arguments['invtau0_nadm_dg'] = 10**(self.cosmo_arguments[elem])
                del self.cosmo_arguments[elem]
            elif elem == 'w0wa':
                self.cosmo_arguments['wa_fld'] = self.cosmo_arguments[elem] - self.cosmo_arguments['w0_fld']
                del self.cosmo_arguments[elem]
            elif elem == 'N_eff_camb':
                #self.cosmo_arguments['N_ur'] = 2./3.*self.cosmo_arguments[elem]
                N_eff_fid = 3.044
                self.cosmo_arguments['N_ur'] = self.cosmo_arguments[elem] - N_eff_fid/3.
                del self.cosmo_arguments[elem]
            elif elem == 'm_nu_camb':
                N_eff_fid = 3.044
                #print ("Treat m_nu_camb")
                #if "N_eff_camb" in self.cosmo_arguments:
                #    gfact = self.cosmo_arguments['N_eff_camb']/3.
                #elif "N_ur" in self.cosmo_arguments:
                #    gfact = self.cosmo_arguments['N_ur']/2.
                #else:
                #    raise ValueError("Could not find Omega_ncdm because N_eff_camb and N_ur both undefined")
                gfact =N_eff_fid/3.

                self.cosmo_arguments['T_ncdm'] = pow(4./11.,1./3.)*pow(gfact,1./4.)
                if "h" in self.cosmo_arguments:
                    h = float(self.cosmo_arguments['h'])
                elif "H0" in self.cosmo_arguments:
                    h = float(self.cosmo_arguments['H0'])/100

                m_nu_camb = float(self.cosmo_arguments['m_nu_camb'])
                self.cosmo_arguments['Omega_ncdm'] = m_nu_camb*gfact**0.75/94.07/h**2
                del self.cosmo_arguments[elem]
            elif elem == 'Omega_m_camb':
                #print("Treat Omega_m_camb")
                if "Omega_ncdm" in self.cosmo_arguments:
                    self.cosmo_arguments['Omega_cdm'] = self.cosmo_arguments['Omega_m_camb'] - self.cosmo_arguments['Omega_b'] - self.cosmo_arguments['Omega_ncdm']
                else:
                    raise ValueError("Could not find Omega_cdm because Omega_ncdm not computed yet")
                del self.cosmo_arguments[elem]
            elif elem == 'S_8':
                # infer sigma8 from S_8, h, omega_b, omega_cdm, and omega_nu (omega_nu = sum(m_nu) / 93.14)
                # Originally by B. Stoelzner for one massive neutrino
                # Generalized by T. Brinckmann to arbitrary number of massive neutrinos and different parameter names
                # applied MCMC parameter scaling only for small omega parameters, betting no one rescales the rest

                # Check for the Hubble parameter, return error if not provided
                if 'h' in self.cosmo_arguments:
                    h = self.cosmo_arguments['h']
                elif 'H0' in self.cosmo_arguments:
                    h = self.cosmo_arguments['H0']/100.
                else:
                    raise io_mp.ConfigurationError("S_8 as a sampling parameter requires h",
                                                   "or H0 as a fixed or varying parameter")
                # Check for omega_b, return error if not provided
                if 'omega_b' in self.cosmo_arguments:
                    scale = 1.
                    if 'omega_b' in self.mcmc_parameters:
                        scale = self.mcmc_parameters['omega_b']['scale']
                    omega_b = self.cosmo_arguments['omega_b'] * scale
                elif 'Omega_b' in self.cosmo_arguments:
                    omega_b = self.cosmo_arguments['Omega_b'] * h**2
                else:
                    raise io_mp.ConfigurationError("S_8 as a sampling parameter requires omega_b",
                                                   "or Omega_b as a fixed or varying parameter")

                # Check for omega_cdm, return error if not provided
                # S_8 is not implemented for different kinds of DM, e.g. DCDM or WDM
                # If your DM behaves as CDM and doesn't affect S_8 beyond that
                # then you can add an if statement below to catch the variable name
                if 'omega_cdm' in self.cosmo_arguments:
                    scale = 1.
                    if 'omega_cdm' in self.mcmc_parameters:
                        scale = self.mcmc_parameters['omega_cdm']['scale']
                    omega_cdm = self.cosmo_arguments['omega_cdm'] * scale
                elif 'Omega_cdm' in self.cosmo_arguments:
                    omega_cdm = self.cosmo_arguments['Omega_cdm'] * h**2
                else:
                    raise io_mp.ConfigurationError("S_8 as a sampling parameter requires omega_cdm",
                                                   "or Omega_cdm as a fixed or varying parameter")

                # Check for massive neutrinos and compute omega_nu if relevant
                if 'omega_ncdm' in self.cosmo_arguments:
                    scale = 1.
                    if 'omega_nu' in self.mcmc_parameters:
                        scale =self.mcmc_parameters['omega_nu']['scale']
                    omega_nu = self.cosmo_arguments['omega_ncdm'] * scale
                elif 'Omega_ncdm' in self.cosmo_arguments:
                    omega_nu = self.cosmo_arguments['omega_ncdm'] * h**2
                elif 'M_tot' in self.cosmo_arguments:
                    omega_nu = self.cosmo_arguments['M_tot'] / 93.14
                elif 'M_nu' in self.cosmo_arguments:
                    omega_nu = self.cosmo_arguments['M_nu'] / 93.14
                elif 'Mnu' in self.cosmo_arguments:
                    omega_nu = self.cosmo_arguments['Mnu'] / 93.14
                elif '{\sum}m_nu' in self.cosmo_arguments:
                    omega_nu = self.cosmo_arguments['{\sum}m_nu'] / 93.14
                elif '{\sum}mnu' in self.cosmo_arguments:
                    omega_nu = self.cosmo_arguments['{\sum}mnu'] / 93.14
                elif 'm_ncdm' in self.cosmo_arguments:
                    # Check number of massive neutrinos, assume 1 (CLASS default) if not specified
                    if 'deg_ncdm' in self.cosmo_arguments:
                        if type(self.cosmo_arguments['deg_ncdm']) == str:
                            deg_ncdm = np.array(self.cosmo_arguments['deg_ncdm'].split(','), dtype='float')
                        else:
                            deg_ncdm = np.array(float(self.cosmo_arguments['deg_ncdm']))
                    else:
                        deg_ncdm = np.array(1.)

                    # Compute neutrino mass sum based on number of massive neutrinos
                    if type(self.cosmo_arguments['m_ncdm']) == str:
                        m_ncdm = np.array(self.cosmo_arguments['m_ncdm'].split(','), dtype='float')
                    else:
                        m_ncdm = np.array(float(self.cosmo_arguments['m_ncdm']))
                    omega_nu = np.dot(deg_ncdm,m_ncdm) / 93.14
                else:
                    omega_nu = 0.
                self.cosmo_arguments['sigma8'] = self.cosmo_arguments['S_8'] * ((0.3*h**2) / (omega_b+omega_cdm+omega_nu))**0.5
                del self.cosmo_arguments[elem]

            # Finally, deal with all the parameters ending with __i, where i is
            # an integer. Replace them all with their name without the trailing
            # double underscore, concatenated with each other. The test is
            # always on the one ending with __1, as it will be the first on the
            # list, and deal with all the others.
            elif re.search(r'__1', elem):
                original_name = re.search(r'(.*)__1', elem).groups()[0]
                # Recover the values of all the other elements
                values = [self.cosmo_arguments[elem]]
                for other_elem in self.get_mcmc_parameters(['cosmo']):
                    match = re.search(r'%s__([2-9])' % original_name,
                                      other_elem)
                    if match:
                        values.append(self.cosmo_arguments[other_elem])
                # create the cosmo_argument
                self.cosmo_arguments[original_name] = ', '.join(
                    ['%g' % value for value in values])
                # Delete the now obsolete entries of the dictionary
                for index in range(1, len(values)+1):
                    del self.cosmo_arguments[
                        original_name + '__%i' % index]
    @staticmethod
    def folder_is_initialised(folder):
        """
        Static method to call for checking if a folder was already initialised

        This method can be used to speed up the mpi initialisation in
        :mod:`run`. If a process finds that the folder is already a proper
        Monte Python one, it sends directly a 'go' signal to its next in line.

        .. warning::

            This method assumes that the last lines of the log.param are the
            path indication. If this would ever change, adjust this method
            accordingly.

        """
        # If the folder is not there, easy answer: False!
        if not os.path.isdir(folder):
            return False
        # Recover the log.param from the folder, and assert it exists
        log_param_path = os.path.join(folder, 'log.param')
        if not os.path.isfile(log_param_path):
            return False
        # Quickly load it to a string, and assert that the path has been
        # written (which are the last lines)
        with open(log_param_path, 'r') as log_param:
            text = log_param.readlines()
            if text[-1].find('path[') != -1:
                return True
            else:
                return False

    def __cmp__(self, other):
        """
        Redefinition of the 'compare' method for two instances of this class.

        It will decide which basic operations to perform when the code asked if
        two instances are the same (in case you want to launch a new chain in
        an existing folder, with your own parameter file) Comparing
        cosmological code versions (warning only, will not fail the comparison)

        """
        if self.version != other.version:
            warnings.warn(
                "You are running with a different version of your " +
                "cosmological code")

        # Defines unordered version of the dictionaries of parameters
        self.uo_parameters = {}
        other.uo_parameters = {}

        # Check if all the experiments are tested again,
        if len(list(set(other.experiments).symmetric_difference(
                set(self.experiments)))) == 0:
            # Check that they have been called with the same .data file, stored
            # in dictionary when initializing.
            for experiment in self.experiments:
                for elem in self.lkl[experiment].dictionary:
                    if self.lkl[experiment].dictionary[elem] != \
                            other.lkl[experiment].dictionary[elem]:
                        sys.stdout.write('in your parameter file: ')
                        print(self.lkl[experiment].dictionary)
                        sys.stdout.write('in log.param:           ')
                        print(other.lkl[experiment].dictionary)
                        return -1
            # Fill in the unordered version of dictionaries
            for key, elem in dictitems(self.mcmc_parameters):
                self.uo_parameters[key] = elem['initial']
            for key, elem in dictitems(other.mcmc_parameters):
                other.uo_parameters[key] = elem['initial']

            # And finally compare them (standard comparison between
            # dictionnaries, will return True if both have the same keys and
            # values associated to them.
            return cmp(self.uo_parameters, other.uo_parameters)
        else:
            return -1

    def __call__(self, ctx):
        """
        Interface layer with CosmoHammer

        Store quantities to a the context, to be accessed by the Cosmo Module
        and each of the likelihoods.

        Parameters
        ----------
        ctx : context
                Contains several dictionaries storing data and cosmological
                information

        """
        # Recover the cosmological parameter value from the context
        parameters = ctx.getParams()

        # Storing them as current points
        for index, elem in enumerate(self.get_mcmc_parameters(["varying"])):
            self.mcmc_parameters[elem]['current'] = parameters[index]

        # Propagating this to the cosmo_arguments dictionary
        self.update_cosmo_arguments()

        # Store itself into the context
        ctx.add('data', self)


class Parameter(dict):
    """
    Store all important fields, and define a few convenience methods

    """
    def __init__(self, array, key):
        """
        This class replaces the old function defined in the Data class, called
        `from_input_to_mcmc_parameters`. The traduction is now done inside the
        Parameter class, which interprets the array given as an input inside
        the parameter file, and returns a dictionary having all relevant fields
        initialized.

        .. warning::

            This used to be an ordered dictionary, for no evident reason. It is
            now reverted back to an ordinary dictionary. If this broke
            anything, it will be reverted back

        At the end of this initialization, every field but one is filled for
        the specified parameter, be it fixed or varying. The missing field is
        the 'last_accepted' one, that will be filled in the module :mod:`mcmc`.

        .. note::

            The syntax of the parameter files is defined here - if one
            wants to change it, one should report the changes in there.

        The other fields are

        Attributes
        ----------
        initial : array
            Initial array of input values defined in the parameter file.
            Contains (in this order) `mean`, `minimum`, `maximum`, `1-sigma`.
            If the min/max values (**TO CHECK** proposal density boundaries)
            are unimportant/unconstrained, use `None` or `-1` (without a period
            !)
        scale : float
            5th entry of the initial array in the parameter file, defines the
            factor with which to multiply the values defined in `initial` to
            give the real value.
        role : str
            6th entry of the initial array, can be `cosmo`, `nuisance` or
            `derived`. A `derived` parameter will not be considered as varying,
            but will be instead recovered from the cosmological code for each
            point in the parameter space.
        prior : :class:`Prior <prior.Prior>`
            defined through the optional 7th entry of the initial array, can be
            ommited or set to `flat` (same), or set to `gaussian`. An instance
            of the :class:`prior` defined in :mod:`prior` will be initialized
            and set to this value.
        tex_name : str
            A tentative tex version of the name, provided by the function
            :func:`io_mp.get_tex_name`.
        status : str
            Depending on the `1-sigma` value in the initial array, it will be
            set to `fixed` or `varying` (resp. zero and non-zero)
        current : float
            Stores the value at the current point in parameter space (`not
            allowed initially`)

        Parameters
        ----------
        value : list
            Array read from the parameter file
        key : str
            Name of the parameter

        """
        # calling the parent method initialization
        dict.__init__(self)

        self['initial'] = array[0:4]
        self['scale'] = array[4]
        self['role'] = array[-1]
        self['tex_name'] = io_mp.get_tex_name(key)
        if array[3] == 0:
            self['status'] = 'fixed'
            self['current'] = array[0]
        else:
            self['status'] = 'varying'
        self['prior'] = prior.Prior(array)


class Container(object):
    """Dummy class to act as a namespace for data"""
    pass


if __name__ == "__main__":
    import doctest
    import shutil
    from initialise import initialise
    folder = os.path.join('tests', 'doc')
    cosmo, data, command_line, _ = initialise('-o %s -p test.param' % folder)
    doctest.testmod(extraglobs={'data': data})
    shutil.rmtree(folder)
