"""
Input-Output handling

Handles all the input/output of the code (at least most of it). If something is
printed that does not satisfy you (number of decimals, for instance, in the
output files), you only have to find the called function and change a number.

Whenever the arguments of the functions are :code:`command_line` or
:code:`data`, no mention of them will be done - as it is now clear. On the
contrary, if there are more arguments, they will be detailled.

This module also defines a new class :class:`File`, that extends
:py:class:`file`, which provides a tail function. It is used in
:func:`sampler.read_args_from_chain`.

Finally, the way the error messages are displayed is set there, along with
ascii-art for the exclamation mark sign.
"""

from __future__ import print_function
import os
import re  # Module to handle regular expressions
from datetime import date
import fcntl
import textwrap  # used to format the error messages
import sys
import io

# Ascii art for error display
START_LINE = {}
START_LINE['error'] = [r' /|\   ',
                       r'/_o_\  ',
                       r'       ']
START_LINE['warning'] = [r' /!\ ',
                         r'     ']
START_LINE['info'] = [r' /!\ ',
                      r'     ']

STANDARD_LENGTH = 80  # standard, increase if you have a big screen


def log_parameters(data, command_line):
    """
    Write the first lines of the log.param

    Writes the beginning of log.param, starting with the header with the
    cosmological code version and potential git hash and branch name, and then
    recopies entirely the input parameter file.

    """
    with open(os.path.join(command_line.folder, 'log.param'), 'w') as log:
        log.write("#-----{0} {1} (branch: {2}, hash: {3})-----\n\n".format(
            data.cosmological_module_name, data.version,
            data.git_branch, data.git_version))
        with open(command_line.param, 'r') as param_file:
            for line in param_file:
                log.write(line)


def log_likelihood_parameters(likelihood, command_line):
    """
    Write down the interpreted .data file of the input likelihood to log.param

    .. warning::

        Since version 2.0.2, the lines are not copied verbatim, they are first
        interpreted, then copied. This allows for overriding of parameters from
        the input.param file.
    """
    with open(os.path.join(command_line.folder, 'log.param'), 'a') as log:
    #tolog = open(likelihood.path, 'r')
        log.write("\n\n#-----Likelihood-{0}-----\n".format(likelihood.name))
        for key, value in dictitems(likelihood.dictionary):
            if type(value) != type(''):
                log.write("%s.%s = %s\n" % (
                    likelihood.name, key, value))
            else:
                log.write("%s.%s = '%s'\n" % (
                    likelihood.name, key, value))
    #for line in tolog:
        #log.write(line)
    #tolog.seek(0)
    #tolog.close()


def log_cosmo_arguments(data, command_line):
    """
    Write down the `cosmo_arguments` used to log.param

    Third function called when writing log.param. It is understood here that
    all the other parameters for the cosmological modules are set to their
    default value directly in the program.

    It is written as an update for the dictionary cosmo_arguments (i.e. as
    :code:`dict.update()` and not as :code:`dict =`) in order not to erase
    previously initialized data.

    """
    if len(data.cosmo_arguments) >= 1:
        log = open(os.path.join(command_line.folder, 'log.param'), 'a')
        log.write('\n\n#-----------Cosmological-arguments---------\n')
        log.write('data.cosmo_arguments.update({0})\n'.format(
            data.cosmo_arguments))
        log.close()


def log_default_configuration(data, command_line):
    """
    Log the .conf file to log.param

    Fourth and last function called when writing log.param. Only useful if you
    have several versions of your cosmological code installed in different
    locations, or different versions of Clik. But, as you never know what might
    go wrong, it is logged everytime !

    TODO: should the root be still logged? (@packaging)

    """
    log = open(os.path.join(command_line.folder, 'log.param'), 'a')
    log.write('\n\n#--------Default-Configuration------\n')
    for key, value in dictitems(data.path):
        log.write("data.path['{0}']\t= '{1}'\n".format(key, value))
    log.close()


def log_parameter_names(data, command_line):
    """
    Log the parameter names to <date_today>_<N>_.paramnames for GetDist compatibility.
    """
    number = command_line.N
    # If N was not provided, assumes N is 10 (default value)
    if not number:
        number = data.N

    # If a restart set number to correspond to new chains filename
    if command_line.restart:
        number += int(
            command_line.restart.split(os.path.sep)[-1].split('__')[0].
            split('_')[1])

    outname_base = '{0}_{1}_'.format(date.today(), number)
    log = open(os.path.join(command_line.folder, outname_base+'.paramnames'), 'w')
    # Create list of varying and derived parameters
    param = data.get_mcmc_parameters(['varying'])
    for elem in data.get_mcmc_parameters(['derived']):
        param.append(elem)
    # New type of parameter, derived_lkl: this is a derived parameter calculated in the likelihood, and not known to CLASS. Added by D. C. Hooper
    for elem in data.get_mcmc_parameters(['derived_lkl']):
        param.append(elem)
    for name in param:
        # Use get_tex_name to convert parameter name to tex name
        tex_name = get_tex_name(name, data.mcmc_parameters[name]['scale'])
        # Remove illegal symbols
        tex_name = re.sub('[$*&]', '', tex_name)
        name = re.sub('[$*&]', '', name)
        log.write("%s \t %s \n" % (name, tex_name))
    log.close()

def print_parameters(out, data):
    """
    Will print the parameter names. In the code, :code:`out` is simply the
    standard output, as this information will not be printed on the output
    file.

    Indeed, you will be able to recover these information from the log.param.

    .. warning::

        Please pay attention to the fact that, once launched, the order of the
        parameters in log.param is crucial, as is it the only place where it is
        stored.

    """
    param = data.get_mcmc_parameters(['varying'])
    for elem in data.get_mcmc_parameters(['derived']):
        param.append(elem)
    # New type of parameter, derived_lkl: this is a derived parameter calculated in the likelihood, and not known to CLASS. Added by D. C. Hooper
    for elem in data.get_mcmc_parameters(['derived_lkl']):
        param.append(elem)
    out.write('\n#  -LogLkl\t')
    for i in range(len(param)):
        if data.mcmc_parameters[param[i]]['scale'] != 1:
            number = data.mcmc_parameters[param[i]]['scale']
            if (number > 100. or number < 0.01):
                string = '%0.e%s' % (1./number, param[i])
            else:
                string = '%0.2g%s' % (1./number, param[i])
        else:
            string = '%s' % param[i]
        out.write("%-16s" % string)
    out.write('\n')


def print_vector(out, N, loglkl, data):
    """
    Print the last accepted values to :code:`out`

    Parameters
    ----------
    out : list
        Array containing both standard output and the output file.

        This way, if you run in interactive mode, you will be able to monitor
        the progress of the chain.
    N : int
        Multiplicity of the point, `i.e.` number of times the code stayed at
        this particular place.
    loglkl : float
        Value of the (- log likelihood) at this point

    .. note::

        It is the `last_accepted` point that is printed, and **not** the
        `current` one (obviously, as one does not know yet the multiplicity of
        the current one !)

    """

    for j in range(len(out)):
        out[j].write('%.4g  %.6g\t' % (N, -loglkl))
        for elem in data.get_mcmc_parameters(['varying']):
            out[j].write('%.6e\t' %
                         data.mcmc_parameters[elem]['last_accepted'])
        for elem in data.get_mcmc_parameters(['derived']):
            out[j].write('%.6e\t' %
                         data.mcmc_parameters[elem]['last_accepted'])
        for elem in data.get_mcmc_parameters(['derived_lkl']):
            out[j].write('%.6e\t' %
                         data.mcmc_parameters[elem]['last_accepted'])
        out[j].write('\n')

def print_error_log(cosmo, data, failure_message):
    folder = os.path.dirname(data.out_name)
    fname = os.path.join(folder, 'error_chains.log')
    with open(fname, 'a+') as out:
        for elem in data.get_mcmc_parameters(['varying']):
            out.write('%.6e\t' %
                         data.mcmc_parameters[elem]['current'])
        out.write(str(int(get_error_code(failure_message))))
        out.write('\n')

    fname = os.path.join(folder, 'error_params.log')
    with open(fname, 'a+') as out:
        out.write(str(cosmo.pars) + '\n')

    fname = os.path.join(folder, 'error_msg.log')
    with open(fname, 'a+') as out:
        out.write(str(cosmo.pars) + '\n')
        out.write(str(failure_message) + '\n')
        out.write('\n')
        out.write(78*'#' + '\n')
        out.write(78*'#' + '\n')
        out.write('\n')

def get_error_code(failure_message):
    """
    Based on github.com/miguelzuma/montepython_zuma/io_mp.py
    print_error function
    """
    error_msg = str(failure_message)
    err_code = -1 # -1 is for unmarked error

    # errors in background are 1-10, perturbations 10-99...
    if 'Shooting failed' in error_msg:
      err_code = 0
    elif all(x in error_msg for x in ['Gradient', 'scalar']):
      err_code = 1
    elif all(x in error_msg for x in ['Gradient', 'tensor']):
      err_code = 2
    elif all(x in error_msg for x in ['Ghost', 'scalar']):
      err_code = 3
    elif all(x in error_msg for x in ['Ghost', 'tensor']):
      err_code = 4
    elif 'Isnan v_X' in error_msg:
      err_code = 10
    elif all(x in error_msg for x in ['early_smg', 'adiabatic']):
      err_code = 11

    return err_code

def refresh_file(data):
    """
    Closes and reopen the output file to write any buffered quantities

    """
    data.out.close()
    data.out = open(data.out_name, 'a')


def create_output_files(command_line, data):
    """
    Automatically create a new name for the chain.

    This routine takes care of organising the folder for you. It will
    automatically generate names for the new chains according to the date,
    number of points chosen.

    .. warning::

        The way these names are generated (with the proper number of _, __, -,
        and their placement) is exploited in the rest of the code in various
        places.  Please keep that in mind if ever you are in the mood of
        changing things here.

    """
    if command_line.restart is None:
        number = command_line.N
    else:
        number = int(
            command_line.restart.split(os.path.sep)[-1].split('__')[0].
            split('_')[1]) + command_line.N

    # output file
    outname_base = '{0}_{1}__'.format(date.today(), number)
    suffix = 0
    trying = True
    if command_line.chain_number is None:
        for files in os.listdir(command_line.folder):
            if files.find(outname_base) != -1:
                if int(files.split('__')[-1].split('.')[0]) > suffix:
                    suffix = int(files.split('__')[-1].split('.')[0])
        suffix += 1
        while trying:
            data.out = open(os.path.join(
                command_line.folder, outname_base)+str(suffix)+'.txt', 'w')
            try:
                lock(data.out, fcntl.LOCK_EX | fcntl.LOCK_NB)
                trying = False
            except LockError:
                suffix += 1
        data.out_name = os.path.join(
            command_line.folder, outname_base)+str(suffix)+'.txt'
        print('Creating %s\n' % data.out_name)
    else:
        data.out_name = os.path.join(
            command_line.folder, outname_base)+command_line.chain_number+'.txt'
        data.out = open(data.out_name, 'w')
        print('Creating %s\n' % data.out_name)
    # in case of a restart, copying the whole thing in the new file
    if command_line.restart is not None:
        # Construct filename of old chain from input.
        # Will only load files with chain file suffix equal to or greater than
        # the one passed with -r flag, e.g. <path>/1969-10-05_100000__1.txt
        # and four MPI processes will load files 1-4 with that chainbase.
        filebase, sep1, filesuffix = command_line.restart.partition('__')
        chainbase, sep2, fileext = filesuffix.partition('.')
        if command_line.chain_number is None:
            restartname = filebase + sep1 + str(int(chainbase)+suffix-1) + sep2 + fileext
        else:
            restartname = filebase + sep1 + command_line.chain_number + sep2 + fileext
        # Copy old chain to the new chain. Return error if file doesn't exist.
        try:
            for line in open(restartname, 'r'):
                data.out.write(line)
        except:
            raise ConfigurationError('Error loading chains files. '
                                     'Ensure number of MPI processes =< number of chains to restart, '
                                     'and chain filename passed with -r <path/filename> corresponds to '
                                     'the full chain name of the lowest chain suffix to restart, e.g. '
                                     '-r <path>/1969-10-05_100000__1.txt to restart from file 1 and '
                                     'subsequent files corresponding to the number of MPI processes.')

def get_tex_name(name, number=1):
    """
    Simplistic tex name transformer.

    Essentially tries to add a backslash in front of known possible greek
    letters, and insert curly brackets { } around statement following an
    _ or a ^. It will also try to include the scale into the name in a nice
    way.

    .. note::

        This might easily fail on simple names, like `beta_plus_lambda`. In
        this case, please use an extra plot file with the command line option
        :code:`-extra plot_file`, or come up with a better function !

    .. note::

        This function returns immediatly with the unmodified name if it already
        contains the LaTeX symbol for math, $.

    Parameters
    ----------
    name : str
        Input name

    Keyword Arguments
    -----------------
    number : float
        Scale

    """
    # First, if the name already contains $ signs, returns it unmodified
    if name.find("$") != -1:
        return name
    tex_greek = ['omega', 'tau', 'alpha', 'beta', 'delta', 'nu',
                 'Omega', 'Lambda', 'lambda', 'Delta', 'mu', 'sigma', 'gamma', 'theta']
    for elem in tex_greek:
        if elem in name:
            position = name.find(elem)
            name = name[:position]+"""\\"""+name[position:]
    if name.find('_') != -1:
        temp_name = name.split('_')[0]+'_{'
        for i in range(1, len(name.split('_'))):
            temp_name += name.split('_')[i]+' '
        name = temp_name + '}'
    if number == 1:
        name = "${0}$".format(name)
    elif number < 1000 and number > 1:
        name = "$%0.d~%s$" % (number, name)
    else:
        temp_name = "$%0.e%s$" % (number, name)
        m = re.search(r'(?:\$[0-9]*e\+[0]*)([0-9]*)(.*)', temp_name)
        sign = '+'
        if m is None:
            m = re.search(r'(?:\$[0-9]*e\-[0]*)([0-9]*)(.*)', temp_name)
            sign = '-'
        name = '$10^{'+sign+m.groups()[0]+'}'+m.groups()[1]
    return name


def write_covariance_matrix(covariance_matrix, names, path):
    """
    Store the covariance matrix to a file
    """
    with open(path, 'w') as cov:
        cov.write('# %s\n' % ', '.join(['%16s' % name for name in names]))

        for i in range(len(names)):
            for j in range(len(names)):
                if covariance_matrix[i][j] > 0:
                    cov.write(' %.6e\t' % covariance_matrix[i][j])
                else:
                    cov.write('%.6e\t' % covariance_matrix[i][j])
            cov.write('\n')

def write_bestfit_file(bestfit, names, path):
    """
    Store the bestfit parameters to a file
    """
    with open(path, 'w') as bestfit_file:
        bestfit_file.write(
            '# %s\n' % ', '.join(['%16s' % name for name in names]))
        # Removing scale factors in order to store true parameter values
        for i in range(len(names)):
            #bfvalue = chain[a[0], 2+i]*info.scales[i, i]
            bf_value = bestfit[i]
            if bf_value > 0:
                bestfit_file.write(' %.6e\t' % bf_value)
            else:
                bestfit_file.write('%.6e\t' % bf_value)
        bestfit_file.write('\n')


def pretty_print(string, status, return_string=False):
    """
    Return the string formatted according to its status

    The input is a potentially long message, describing the problem.
    According to the severity of its status (so far, 'error' will exit the
    program, whereas 'warning' and 'info' will go through anyway).

    Standard length has been defined globally, as well as the ascii-art
    dictionary of arrays START_LINE.

    """

    if return_string:
        output = ''
    length = STANDARD_LENGTH-len(START_LINE[status][0])
    # Remove unwanted spaces (coming from carriage returns in the input string)
    # and handle voluntary carriage returns specified with \n
    first_cleanup = [' '.join(elem.lstrip(' ').split())
                     for elem in string.split('\n')]
    splitted = []
    # Recover the lines splitted at correct length
    for elem in first_cleanup:
        splitted.extend(textwrap.wrap(elem, length))

    if status == 'error':
        # Add a blank line so that the error displays better
        if return_string:
            output += '\n'
        else:
            print

    # Add in front the appropriate fancy display
    index = 0
    for line in splitted:
        # If the number of needed lines is bigger than the ascii-art, the last
        # line of it (empty) will be used.
        if index < len(START_LINE[status]):
            start_index = index
        else:
            start_index = len(START_LINE[status])-1
        if return_string:
            output += START_LINE[status][start_index]+line+'\n'
        else:
            print(START_LINE[status][start_index]+line)
        index += 1
    if return_string:
        return output
    else:
        return


def safe_exec(string):
    """Attempt at executing a string from file in a secure way"""
    exec(string, {'__builtins__': {}})


class File(io.FileIO):
    """
    New class of file, to provide an equivalent of the tail command (on linux).

    It will be used when starting from an existing chain, and avoids circling
    through an immense file.
    """

    def tail(self, lines_2find=1):
        """Imitates the classic tail command"""
        self.seek(0, 2)     # go to end of file
        bytes_in_file = self.tell()
        lines_found, total_bytes_scanned = 0, 0
        while (lines_2find+1 > lines_found and
                bytes_in_file > total_bytes_scanned):
            byte_block = min(1024, bytes_in_file-total_bytes_scanned)
            self.seek(-(byte_block+total_bytes_scanned), 2)
            total_bytes_scanned += byte_block
            lines_found += self.read(1024).count('\n')
        self.seek(-total_bytes_scanned, 2)
        line_list = list(self.readlines())
        return line_list[-lines_2find:]


class LockError(Exception):
    """
    .. warning::

        in the process of being tested
    """
    # Error codes:
    LOCK_FAILED = 1


def lock(file_obj, flags):
    """
    Lock a given file to prevent other instances of the code to write to the
    same file.

    .. warning::

        in the process of being tested

    """
    import fcntl
    try:
        fcntl.flock(file_obj.fileno(), flags)
    except IOError as exc_value:
        # The exception code varies on different systems so we'll catch
        # every IO error
        raise LockError(*exc_value)


def unlock(file_obj):
    """
    Unlock a previously locked file.

    .. warning::

        in the process of being tested

    """
    import fcntl
    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


def warning_message(message, *args):
    """
    Custom implementation of `showwarning` from :mod:`warnings`

    """
    pretty_print(message.args[0], "warning")


class MyError(Exception):
    """
    Base class defining the general presentation of error messages

    """
    def __init__(self, message):
        """Reformat the name of the class for easier reading"""
        Exception.__init__(self)
        self.message = message
        name = self.__class__.__name__
        self.name = ''
        # Extract the name, and add spaces between the capital letters
        for index, letter in enumerate(name):
            if letter.isupper():
                if index > 0:
                    self.name += ' ' + letter
                else:
                    self.name += letter
            else:
                self.name += letter

    def __str__(self):
        """Define the behaviour under the print statement"""
        return '\n\n' + self.name + ':' + pretty_print(
            self.message, "error", True)


class CosmologicalModuleError(MyError):
    """For all problems linked to the cosmological module"""
    pass


class ConfigurationError(MyError):
    """Missing files, libraries, etc..."""
    pass


class MissingLibraryError(MyError):
    """Missing Cosmo module, Planck, ..."""
    pass


class LikelihoodError(MyError):
    """Problems when computing likelihood, missing nuisance, etc..."""
    pass


class FiducialModelWritten(MyError):
    """Used to exit the code in case of writing a fiducial file"""
    pass


class AnalyzeError(MyError):
    """Used when encountering a fatal mistake in analyzing chains"""
    pass

# Nils : Added for python 3.x and python 2.x compatibility
def dictitems(d):
    if sys.version_info >= (3,0):
        return d.items()
    else:
        return d.iteritems()

def dictkeys(d):
    if sys.version_info >= (3,0):
        return d.keys()
    else:
        return d.iterkeys()

def dictvalues(d):
    if sys.version_info >= (3,0):
        return d.values()
    else:
        return d.itervalues()
