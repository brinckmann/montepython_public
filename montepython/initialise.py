"""
.. module:: initialise
    :synopsis: intialisation
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

"""
from __future__ import print_function
import io_mp
import parser_mp   # parsing the input command line
from data import Data
import sys
import os
from io_mp import dictitems,dictvalues,dictkeys

def initialise(custom_command=''):
    """
    Initialisation routine

    This function recovers the input from the command line arguments, from
    :mod:`parser_mp`, the parameter files.

    It then extracts the path of the used Monte Python code, and proceeds to
    initialise a :class:`data` instance, a cosmological code instance.

    Parameters
    ----------
        custom_command: str
            allows for testing the code
    """
    # Parsing line argument
    command_line = parser_mp.parse(custom_command)

    # Recovering the local configuration
    path = recover_local_path(command_line)

    # check for MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        # set all chains to master if no MPI
        rank = 0

    # Recover Monte Python's version number
    version_path = os.path.join(
        path['root'], 'VERSION')
    with open(version_path, 'r') as version_file:
        version = version_file.readline()
    if not command_line.silent and not rank:
        print('Running Monte Python v%s' % version)

    # If the info flag was used, read a potential chain (or set of chains) to
    # be analysed with default procedure. If the argument is a .info file, then
    # it will extract information from it (plots to compute, chains to analyse,
    # etc...)
    if command_line.subparser_name == "info":
        from analyze import analyze  # only invoked when analyzing
        analyze(command_line)
        return None, None, command_line, False

    # Fill in data, starting from  parameter file. If output folder already
    # exists, the input parameter file was automatically replaced by the
    # existing log.param. This prevents you to run different things in a same
    # folder.
    else:
        data = Data(command_line, path)

        # Overwrite arguments from parameter file with the command line
        if command_line.N is None:
            try:
                command_line.N = data.N
            except AttributeError:
                raise io_mp.ConfigurationError(
                    "You did not provide a number of steps, neither via " +
                    "command line, nor in %s" % command_line.param)

        # Loading up the cosmological backbone. For the moment, only CLASS has been
        # wrapped.
        cosmo = recover_cosmological_module(data)

        # Initialising the sampler
        # MH: Creating the file that will contain the chain
        if command_line.method == 'MH':
            io_mp.create_output_files(command_line, data)
        # NS: Creating the NS subfolder and the MultiNest arguments
        elif command_line.method == 'NS':
            from MultiNest import initialise as initialise_mn
            initialise_mn(cosmo, data, command_line)
        elif command_line.method == 'UN':
            from UltraNest import initialise as initialise_un
            initialise_un(cosmo, data, command_line)
        elif command_line.method == 'SL':
            from SnowLine import initialise as initialise_sl
            initialise_sl(cosmo, data, command_line)
        # PC: Creating the PC subfolder and the PolyChord arguments
        # PC: Creating the PC subfolder and the PolyChord arguments
        elif command_line.method == 'PC':
            from PolyChord import initialise as initialise_pc
            initialise_pc(cosmo, data, command_line)

        return cosmo, data, command_line, True


def recover_local_path(command_line):
    """
    Read the configuration file, filling a dictionary

    Returns
    -------
    path : dict
        contains the absolute path to the location of the code, the data, the
        cosmological code, and potential likelihood codes (clik for Planck,
        etc)
    """
    # Define the dictionnary that will hold the local configuration
    path = {}

    # The path is recovered by taking the path to this file (MontePython.py).
    # By default, then, the data folder is located in the same root directory.
    # Any setting in the configuration file will overwrite this one.
    path['root'] = os.path.sep.join(
        os.path.abspath(__file__).split(os.path.sep)[:-2])
    path['MontePython'] = os.path.join(path['root'], 'montepython')
    path['data'] = os.path.join(path['root'], 'data')

    # the rest is important only when running the MCMC chains
    if command_line.subparser_name == 'run':
        # Configuration file, defaulting to default.conf in your root
        # directory.  This can be changed with the command line option --conf.
        # All changes will be stored into the log.param of your folder, and
        # hence will be reused for an ulterior run in the same directory
        conf_file = os.path.abspath(command_line.config_file)
        if os.path.isfile(conf_file):
            for line in open(conf_file):
                exec(line)
            for key, value in dictitems(path):
                path[key] = os.path.normpath(os.path.expanduser(value))
        else:
            # The error is ignored if reading from a log.param, because it is
            # stored
            if command_line.param.find('log.param') == -1:
                raise io_mp.ConfigurationError(
                    "You must provide a valid  .conf file (I tried to read"
                    "%s) " % os.path.abspath(command_line.config_file) +
                    " that specifies the correct locations for your data "
                    "folder, Class, (Clik), etc...")

    return path


def recover_cosmological_module(data):
    """
    From the cosmological module name, initialise the proper Boltzmann code

    .. note::

        Only CLASS is currently wrapped, but a python wrapper of CosmoMC should
        enter here.

    """
    # Importing the python-wrapped CLASS from the correct folder, defined in
    # the .conf file, or overwritten at this point by the log.param.
    # If the cosmological code is CLASS, do the following to import all
    # relevant quantities
    if data.cosmological_module_name == 'CLASS':
        try:
            classy_path = ''
            for elem in os.listdir(os.path.join(
                    data.path['cosmo'], "python", "build")):
                if elem.find("lib.") != -1:
                    classy_path = os.path.join(
                        data.path['cosmo'], "python", "build", elem)
                if len(classy_path)==1:
                    classy_path = classy_path[0]
                else:
                    stringcheck = "%i.%i"%(sys.version_info.major,sys.version_info.minor)
                    for path in classy_path:
                        if stringcheck in path:
                            classy_path = path
                            break
        except OSError:
            raise io_mp.ConfigurationError(
                "You probably did not compile the python wrapper of CLASS. " +
                "Please go to /path/to/class/python/ and do\n" +
                "..]$ python setup.py build")

        # Inserting the previously found path into the list of folders to
        # search for python modules.
        sys.path.insert(1, classy_path)
        try:
            from classy import Class
        except ImportError:
            raise io_mp.MissingLibraryError(
                "You must have compiled the classy.pyx file. Please go to " +
                "/path/to/class/python and run the command\n " +
                "python setup.py build")

        cosmo = Class()
    else:
        raise io_mp.ConfigurationError(
            "Unrecognised cosmological module. " +
            "Be sure to define the correct behaviour in MontePython.py " +
            "and data.py, to support a new one.")

    return cosmo
