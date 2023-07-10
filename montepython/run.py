"""
.. module:: run_mp
    :synopsis: Run the code, initialising everything and calling the sampler
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

"""
from __future__ import print_function
from initialise import initialise
from data import Data
import io_mp
import sys
import warnings
import os
import re


def run(custom_command=''):
    """
    Main call of the function

    It recovers the initialised instances of cosmo Class, :class:`Data` and the
    NameSpace containing the command line arguments, feeding into the sampler.

    Parameters
    ----------
        custom_command: str
            allows for testing the code
    """
    # Create all the instances of the needed classes to run the code. The safe
    # initialisation handles the errors.
    cosmo, data, command_line, success = safe_initialisation(
        custom_command)

    # If success is False, it means either that the initialisation was not
    # successful, or that it was simply an analysis call. The run should
    # stop
    if not success:
        return

    # Once that the initialisation phase is done, one can import the
    # sampler
    import sampler

    # Generic sampler call
    sampler.run(cosmo, data, command_line)

    return


def mpi_run(custom_command=""):
    """
    Launch a simple MPI run, with no communication of covariance matrix

    Each process will make sure to initialise the folder if needed. Then and
    only then, it will send the signal to its next in line to proceed. This
    allows for initialisation over an arbitrary cluster geometry (you can have
    a single node with many cores, and all the chains living there, or many
    nodes with few cores). The speed loss due to the time spend checking if the
    folder is created should be negligible when running decently sized chains.

    Each process will send the number that it found to be the first available
    to its friends, so that the gathering of information post-run is made
    easier. If a chain number is specified, this will be used as the first
    number, and then incremented afterwards with the rank of the process.
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    success = True
    folder = ''

    # If the process is not the zeroth one, then wait for a signal from your
    # n-1 before initializing the folder
    if rank != 0:
        status = comm.recv(source=rank-1, tag=1)
        folder = comm.recv(source=rank-1, tag=2)
        if status == 'failed':
            success = False
        else:
            number = status

    if success:
        if not custom_command:
            custom_command = " ".join(sys.argv[1:])
        if rank != 0:
            custom_command += " --chain-number %s" % str(int(number)+1)

        # First check if the folder is there
        already_sent = False
        if rank != 0 and rank < nprocs-1:
            status = int(number)+1
            if Data.folder_is_initialised(folder):
                comm.send(status, dest=rank+1, tag=1)
                comm.send(folder, dest=rank+1, tag=2)
                already_sent = True

        # Then, properly initialise
        cosmo, data, command_line, success = safe_initialisation(
            custom_command, comm, nprocs)

        # The first initialisation should check a few more things
        if rank == 0:
            # Check that the run asked is compatible with mpirun and prepare.
            if command_line.subparser_name == 'info':
                warnings.warn(
                    "Analyzing the chains is not supported in mpirun"
                    " so this will run on one core only.")
                status = 'failed'
            elif command_line.method == "MH":
                regexp = re.match(".*__(\w*).txt", data.out_name)
                suffix = regexp.groups()[0]
                status = suffix
            elif command_line.method in ("NS", "PC", "UN", "SL"):
                status = 1
            else:
                warnings.warn(
                    "The method '%s' is not supported"%(command_line.method) +
                    " in mpirun so this will run on one core only.")
                status = 'failed'
            folder = data.out_name
        elif rank < nprocs-1:
            status = int(number)+1
        # Send an "OK" signal to the next in line, giving the its own chain
        # number for the other to add 1
        if rank < nprocs-1:
            if not already_sent:
                comm.send(status, dest=rank+1, tag=1)
                comm.send(folder, dest=rank+1, tag=2)
    else:
        if rank < nprocs-1:
            comm.send('failed', dest=rank+1, tag=1)
            comm.send('', dest=rank+1, tag=2)

    if success:
        import sampler
        sampler.run(cosmo, data, command_line)

    return


def mock_update_run(custom_command=""):
    """
    Tentative covmat update run

    Not reachable yet by any option.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # store the command_line space
    if not custom_command:
        custom_command = " ".join(sys.argv[1:])

    # Do a first run
    mpi_run(custom_command)

    # Compute the covariance matrix
    if rank == 0:
        info_command = from_run_to_info(custom_command)
        initialise(info_command)

    # Make sure that the covariance matrix used is the one just computed
    with_covmat = add_covariance_matrix(custom_command)

    mpi_run(with_covmat)
    return


def safe_initialisation(custom_command="", comm=None, nprocs=1):
    """
    Wrapper around the init function to handle errors

    KeyWord Arguments
    -----------------
    custom_command : str
        testing purposes
    comm : MPI.Intracomm
        object that helps communicating between the processes
    nprocs : int
        number of processes
    """
    try:
        cosmo, data, command_line, success = initialise(custom_command)
    except io_mp.ConfigurationError as message:
        if comm:
            for index in range(1, nprocs):
                comm.send('failed', dest=index, tag=1)
        print(str(message))
        raise io_mp.ConfigurationError(
            "The initialisation was not successful, resulting in a "
            "potentially half created `log.param`. Please see the "
            "above error message. If you run the exact same command, it"
            " will not work. You should solve the problem, and try again.")
    except KeyError as e:
        if comm:
            for index in range(1, nprocs):
                comm.send('failed', dest=index, tag=1)
        raise io_mp.ConfigurationError(
            "You are running in a folder that was created following "
            "a non-successful initialisation (wrong parameter name, "
            "wrong likelihood, etc...). If you have solved the issue, you "
            "should remove completely the output folder, and try again." +
            " Alternatively, there could be a problem with "+str(e))
    return cosmo, data, command_line, success


def from_run_to_info(command):
    """
    Translate a command corresponding to a run into one for analysis

    """
    original = command.split()
    new = ['info']

    # Recover the folder
    index = original.index('-o')
    folder = original[index+1]

    # Recover chain number
    index = original.index('-N')
    number = original[index+1]

    # Recover all relevant chains
    chains = [os.path.join(folder, elem) for elem in os.listdir(folder)
              if elem.find(number) != -1]
    new.extend(chains)

    # Do not plot the pdf
    new.append("--noplot")
    return " ".join(new)


def add_covariance_matrix(command):
    """
    Make sure that the command uses the covariance matrix from the folder

    """
    original = command.split()

    # recover the folder
    index = original.index('-o')
    folder = original[index+1]
    # get the name of the folder
    name = folder.split(os.path.sep)[-1]
    covname = os.path.join(
        folder, name+'.covmat')
    bfname = os.path.join(
        folder, name+'.bestfit')

    if '-c' in original:
        index = original.index('-c')
        original[index+1] = covname
    else:
        original.extend(['-c', covname])

    if '-b' in original:
        index = original.index('-b')
        original[index+1] = bfname
    elif '--bestfit' in original:
        index = original.index('--bestfit')
        original[index+1] = bfname
    else:
        original.extend(['-b', bfname])

    # Change the number of steps asked
    index = original.index('-N')
    number = list(str(original[index+1]))
    number[0] = str(int(number[0])+1)
    newnumber = "".join(number)
    original[index+1] = str(newnumber)
    return " ".join(original)
