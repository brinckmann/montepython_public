"""
.. module:: SnowLine
    :synopsis: Interface the snowline Laplace approximation and importance sampler with Monte Python

This implementation is similar to the MultiNest interface.

Snowline is a Bayesian sampler based on the Laplace approximation 
and importance sampling optimized with variational Bayes documented at.
https://johannesbuchner.github.io/snowline/

.. moduleauthor:: Johannes Buchner <johannes.buchner.acad@gmx.com>
"""
from __future__ import print_function
#from pymultinest import run as nested_run
import numpy as np
import os
from copy import copy
import warnings
import io_mp
import sampler
import matplotlib.pyplot as plt
import json
import logging
import sys

# Data on file names and MultiNest options, that may be called by other modules

# MultiNest subfolder and name separator
NS_subfolder    = 'SL'
NS_separator    = '-'
# MultiNest file names ending, i.e. after the defined 'base_name'
name_rejected   = NS_separator + 'ev.dat'                 # rejected points
name_post       = NS_separator + '.txt'                   # accepted points
name_post_sep   = NS_separator + 'post_separate.dat'      # accepted points separated by '\n\n'
name_post_equal = NS_separator + 'post_equal_weights.dat' # some acc. points, same sample prob.
name_stats      = NS_separator + 'stats.dat'              # summarized information, explained
name_summary    = NS_separator + 'summary.txt'            # summarized information
# New files
name_paramnames = '.paramnames'            # in the NS/ subfolder
name_arguments  = '.arguments'             # in the NS/ subfolder
name_chain_acc  = 'chain_NS__accepted.txt' # in the chain root folder
name_chain_rej  = 'chain_NS__rejected.txt' # in the chain root folder
# Log.param name (ideally, we should import this one from somewhere else)
name_logparam = 'log.param'

# Multinest option prefix
NS_prefix       = 'SL_'
# User-defined arguments of PyMultiNest, and 'argparse' keywords
# First: basic string -> bool type conversion:
str2bool = lambda s: True if s.lower() == 'true' else False
NS_user_arguments = {
    # General sampling options
    #'n_live_points':
    #    {'help': 'Number of live samples',
    #     'type': int},
    }
# Automatically-defined arguments of PyMultiNest, type specified
NS_auto_arguments = {
    'n_dims':   {'type': int},
    'n_params': {'type': int},
    'verbose':  {'type': str2bool},
    'outputfiles_basename': {'type': str},
    'init_MPI': {'type': str2bool}
    }


def initialise(cosmo, data, command_line):
    """
    Main call to prepare the information for the MultiNest run.
    """

    print("init called!")
    # Convenience variables
    varying_param_names = data.get_mcmc_parameters(['varying'])
    derived_param_names = data.get_mcmc_parameters(['derived'])

    # Check that all the priors are flat and that all the parameters are bound
    is_flat, is_bound = sampler.check_flat_bound_priors(
        data.mcmc_parameters, varying_param_names)
    if not is_flat:
        raise io_mp.ConfigurationError(
            'Nested Sampling with MultiNest is only possible with flat ' +
            'priors. Sorry!')
    if not is_bound:
        raise io_mp.ConfigurationError(
            'Nested Sampling with MultiNest is only possible for bound ' +
            'parameters. Set reasonable bounds for them in the ".param"' +
            'file.')

    # If absent, create the sub-folder NS
    NS_folder = os.path.join(command_line.folder, NS_subfolder)
    if not os.path.exists(NS_folder):
        os.makedirs(NS_folder)

    # Use chain name as a base name for MultiNest files
    chain_name = [a for a in command_line.folder.split(os.path.sep) if a][-1]
    base_name = os.path.join(NS_folder, chain_name)
    #FK: add base folder name to NS_arguments for later reference
    data.NS_arguments['base_dir'] = NS_folder

    # Prepare arguments for PyMultiNest
    # -- Automatic arguments
    data.NS_arguments['n_dims'] = len(varying_param_names)
    data.NS_arguments['n_params'] = (len(varying_param_names) +
                                     len(derived_param_names))
    data.NS_arguments['verbose'] = True
    data.NS_arguments['outputfiles_basename'] = base_name + NS_separator
    # -- User-defined arguments
    for arg in NS_user_arguments:
        value = getattr(command_line, NS_prefix+arg)
        # Special case: clustering parameters
        if arg == 'clustering_params':
            clustering_param_names = value if value != -1 else []
            continue
        # Rest of the cases
        if value != -1:
            data.NS_arguments[arg] = value
        # else: don't define them -> use PyMultiNest default value

    # Clustering parameters -- reordering to put them first
    NS_param_names = []
    for param in varying_param_names:
        if not param in NS_param_names:
            NS_param_names.append(param)
    data.NS_param_names = NS_param_names
            
    # Caveat: multi-modal sampling OFF by default; if requested, INS disabled
    try:
        if data.NS_arguments['multimodal']:
            data.NS_arguments['importance_nested_sampling'] = False
            warnings.warn('Multi-modal sampling has been requested, ' +
                          'so Importance Nested Sampling has been disabled')
    except KeyError:
        data.NS_arguments['multimodal'] = False

    # MPI: don't initialise it inside MultiNest.
    # Rather, it is either initialised by Monte Python (if MPI used) or ignored
    data.NS_arguments['init_MPI']=False

    # Write the MultiNest arguments and parameter ordering
    with open(base_name+name_arguments, 'w') as afile:
        for arg in data.NS_arguments:
            if arg != 'n_clustering_params':
                afile.write(' = '.join(
                    [str(arg), str(data.NS_arguments[arg])]))
            else:
                afile.write('clustering_params = ' +
                            ' '.join(clustering_param_names))
            afile.write('\n')
    with open(base_name+name_paramnames, 'w') as pfile:
        pfile.write('\n'.join(NS_param_names+derived_param_names))


def run(cosmo, data, command_line):
    """
    Main call to run the MultiNest sampler.

    Note the unusual set-up here, with the two following functions, `prior` and
    `loglike` having their docstrings written in the encompassing function.
    This trick was necessary as MultiNest required these two functions to be
    defined with a given number of parameters, so we could not add `data`. By
    defining them inside the run function, this problem was by-passed.

    .. function:: prior

        Generate the prior function for MultiNest

        It should transform the input unit cube into the parameter cube. This
        function actually wraps the method :func:`map_from_unit_interval()
        <prior.Prior.map_from_unit_interval>` of the class :class:`Prior
        <prior.Prior>`.

        Parameters
        ----------
        cube : array
            Contains the current point in unit parameter space that has been
            selected within the MultiNest part.
        ndim : int
            Number of varying parameters
        nparams : int
            Total number of parameters, including the derived ones (not used,
            so hidden in `*args`)


    .. function:: loglike

        Generate the Likelihood function for MultiNest

        Parameters
        ----------
        cube : array
            Contains the current point in the correct parameter space after
            transformation from :func:`prior`.
        ndim : int
            Number of varying parameters
        nparams : int
            Total number of parameters, including the derived ones (not used,
            so hidden in `*args`)

    """
    # Convenience variables
    #derived_param_names = data.get_mcmc_parameters(['derived'])
    NS_param_names      = data.NS_param_names
    n_dim = len(NS_param_names)
    #n_dim_tot = len(NS_param_names) + len(derived_param_names)

    # Function giving the prior probability
    def prior(cube):
        """
        Please see the encompassing function docstring

        """
        params = np.zeros(n_dim)
        
        for i, name in enumerate(NS_param_names):
            params[i] = data.mcmc_parameters[name]['prior']\
                .map_from_unit_interval(cube[i])
        
        # Propagate the information towards the cosmo arguments
        #data.update_cosmo_arguments()
        #for i, name in enumerate(derived_param_names):
        #    cube[ndim+i] = data.mcmc_parameters[name]['current']
        
        return params



    # Function giving the likelihood probability
    def loglike(params):
        """
        Please see the encompassing function docstring

        """
        # Updates values: cube --> data
        for i, name in enumerate(NS_param_names):
            data.mcmc_parameters[name]['current'] = params[i]
        # Propagate the information towards the cosmo arguments
        data.update_cosmo_arguments()
        try:
            lkl = sampler.compute_lkl(cosmo, data)
            print('%.1f' % lkl, params)
        except ValueError:
            lkl = -1e300
        return lkl
    
    #FK: recover name of base folder and remove entry from dict before passing it
    # on to MN:
    base_dir = data.NS_arguments['base_dir']
    #base_dir = 'logs/'
    print("parameters varied:", NS_param_names)
    del data.NS_arguments['base_dir']

    # Launch MultiNest, and recover the output code
    #output = nested_run(loglike, prior, **data.NS_arguments)
    logger = logging.getLogger('snowline')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    handler = logging.FileHandler(os.path.join(base_dir, 'debug.log'))
    formatter = logging.Formatter('%(asctime)s [{}] [%(levelname)s] %(message)s'.format('snowline'),
        datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # if it is new, register to write to stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[{}] %(message)s'.format('snowline'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    from snowline import ReactiveImportanceSampler
    sampler_sl = ReactiveImportanceSampler(NS_param_names, loglike, transform=prior)
    r = sampler_sl.run()
    sampler_sl.print_results()
    np.savetxt(base_dir + '/equal_weighted_samples.txt', r['samples'])
    j = dict(**r)
    del j['samples']
    json.dump(j, open(base_dir + '/info.json', 'w'))
    
    
    from getdist import MCSamples, plots

    samples_g = MCSamples(samples=sampler.results['samples'],
                           names=sampler.results['paramnames'],
                           label='Gaussian',
                           settings=dict(smooth_scale_2D=3), sampler='nested')

    mcsamples = [samples_g]
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.num_plot_contours = 3
    g.triangle_plot(mcsamples, filled=False, contour_colors=plt.cm.Set1.colors)
    plt.savefig(base_dir + '/posterior.pdf', bbox_inches='tight')
    plt.close()
    #output = nested_run(loglike, prior, **data.NS_arguments)

    # Assuming this worked, i.e. if output is `None`,
    # state it and suggest the user to analyse the output.
    output = None
    if output is None:
        # FK: write out the warning message below also as a file in the NS-subfolder
        # so that there's a clear indication for convergence instead of just looking at
        # the STDOUT-log!
        text = 'The sampling with UltraNest is done.\n' + \
               'You can now analyse the output calling Monte Python ' + \
               'with the -info flag in the chain_name/NS subfolder,' + \
               ' or, if you used multimodal sampling, in the ' + \
               'chain_name/mode_# subfolders.'
        fname = os.path.join(base_dir, 'convergence.txt')
        with open(fname, 'w') as afile:
            afile.write(text)


def from_NS_output_to_chains(folder):
    """
    Translate the output of MultiNest into readable output for Monte Python

    This routine will be called by the module :mod:`analyze`.

    If mode separation has been performed (i.e., multimodal=True), it creates
    'mode_#' subfolders containing a chain file with the corresponding samples
    and a 'log.param' file in which the starting point is the best fit of the
    nested sampling, and the same for the sigma. The minimum and maximum value
    are cropped to the extent of the modes in the case of the parameters used
    for the mode separation, and preserved in the rest.

    The mono-modal case is treated as a special case of the multi-modal one.

    """
    chain_name = [a for a in folder.split(os.path.sep) if a][-2]
    base_name = os.path.join(folder, chain_name)

    # Read the arguments of the NS run
    # This file is intended to be machine generated: no "#" ignored or tests
    # done
    NS_arguments = {}
    with open(base_name+name_arguments, 'r') as afile:
        for line in afile:
            arg   = line.split('=')[0].strip()
            value = line.split('=')[1].strip()
            # FK: we need to take care of newly introduced 'base_dir'
            # variable here (by just skipping it):
            if not arg == 'base_dir':
                arg_type = (NS_user_arguments[arg]['type']
                            if arg in NS_user_arguments else
                            NS_auto_arguments[arg]['type'])
                value = arg_type(value)
                if arg == 'clustering_params':
                    value = [a.strip() for a in value.split()]
                NS_arguments[arg] = value
    multimodal = NS_arguments.get('multimodal')
    # Read parameters order
    NS_param_names = np.loadtxt(base_name+name_paramnames, dtype='str').tolist()
    # In multimodal case, if there were no clustering params specified, ALL are
    if multimodal and not NS_arguments.get('clustering_params'):
        NS_arguments['clustering_params'] = NS_param_names

    # Extract the necessary information from the log.param file
    # Including line numbers of the parameters
    with open(os.path.join(folder, '..', name_logparam), 'r') as log_file:
        log_lines = log_file.readlines()
    # Number of the lines to be changed
    param_names = []
    param_lines = {}
    param_data  = {}
    pre, pos = 'data.parameters[', ']'
    for i, line in enumerate(log_lines):
        if pre in line:
            if line.strip()[0] == '#':
                continue

            # These lines allow MultiNest to deal with fixed nuisance parameters 
            sigma = float(line.split(',')[3].strip())
            if sigma == 0.0:
                #If derived parameter, keep it, else discard it:                                 
                paramtype = line.split(',')[5].strip()[1:-2]
                if paramtype != 'derived':
                    continue

            param_name = line.split('=')[0][line.find(pre)+len(pre):
                                            line.find(pos)]
            param_name = param_name.replace('"','').replace("'",'').strip()
            param_names.append(param_name)
            param_data[param_name] = [a.strip() for a in
                                      line.split('=')[1].strip('[]').split(',')]
            param_lines[param_name] = i

    # Create the mapping from NS ordering to log.param ordering
    columns_reorder = [NS_param_names.index(param) for param in param_names]

    # Open the 'stats.dat' file to see what happened and retrieve some info
    stats_file = open(base_name+name_stats, 'r')
    lines = stats_file.readlines()
    stats_file.close()
    # Mode-separated info
    i = 0
    n_modes = 0
    stats_mode_lines = {0: []}
    for line in lines:
        if 'Nested Sampling Global Log-Evidence' in line:
            global_logZ, global_logZ_err = [float(a.strip()) for a in
                                            line.split(':')[1].split('+/-')]
        if 'Total Modes Found' in line:
            n_modes = int(line.split(':')[1].strip())
        if line[:4] == 'Mode':
            i += 1
            stats_mode_lines[i] = []
        # This stores the info of each mode i>1 in stats_mode_lines[i] and in
        # i=0 the lines previous to the modes, in the multi-modal case or the
        # info of the only mode, in the mono-modal case
        stats_mode_lines[i].append(line)
    assert n_modes == max(stats_mode_lines.keys()), (
        'Something is wrong... (strange error n.1)')

    # Prepare the accepted-points file -- modes are separated by 2 line breaks
    accepted_name = base_name + (name_post_sep if multimodal else name_post)
    with open(accepted_name, 'r') as accepted_file:
        mode_lines = [a for a in ''.join(accepted_file.readlines()).split('\n\n')
                      if a != '']
    if multimodal:
        mode_lines = [[]] + mode_lines
    assert len(mode_lines) == 1+n_modes, 'Something is wrong... (strange error n.2)'

# TODO: prepare total and rejected chain

    # Process each mode:
    ini = 1 if multimodal else 0
    for i in range(ini, 1+n_modes):
        # Create subfolder
        if multimodal:
            mode_subfolder = 'mode_'+str(i).zfill(len(str(n_modes)))
        else:
            mode_subfolder = ''
        mode_subfolder = os.path.join(folder, '..', mode_subfolder)
        if not os.path.exists(mode_subfolder):
            os.makedirs(mode_subfolder)

        # Add ACCEPTED points
        mode_data = np.array(mode_lines[i].split(), dtype='float64')
        columns = 2+NS_arguments['n_params']
        mode_data = mode_data.reshape([mode_data.shape[0]//columns, columns])
        # Rearrange: sample-prob | -2*loglik | params (clustering first)
        #       ---> sample-prob |   -loglik | params (log.param order)
        mode_data[:, 1]  = mode_data[:, 1] / 2.
        mode_data[:, 2:] = mode_data[:, [2+j for j in columns_reorder]]
        np.savetxt(os.path.join(mode_subfolder, name_chain_acc),
                   mode_data, fmt='%.6e')

        # If we are not in the multimodal case, we are done!
        if not multimodal:
            break
        # In the multimodal case, we want to write a log.param for each mod
        this_log_lines = copy(log_lines)

        # Get the necessary info of the parameters:
        #  -- max_posterior (MAP), sigma  <---  stats.dat file
        for j, line in enumerate(stats_mode_lines[i]):
            if 'Sigma' in line:
                line_sigma = j+1
            if 'MAP' in line:
                line_MAP = j+2
        MAPs   = {}
        sigmas = {}
        for j, param in enumerate(NS_param_names):
            n, MAP = stats_mode_lines[i][line_MAP+j].split()
            assert int(n) == j+1,  'Something is wrong... (strange error n.3)'
            MAPs[param] = MAP
            n, mean, sigma = stats_mode_lines[i][line_sigma+j].split()
            assert int(n) == j+1,  'Something is wrong... (strange error n.4)'
            sigmas[param] = sigma
        #  -- minimum rectangle containing the mode (only clustering params)
        mins = {}
        maxs = {}
        for param in NS_arguments['clustering_params']:
            # Notice that in the next line we use param_names and not
            # NS_param_names: the chain lines have already been reordered
            values = mode_data[:, 2+param_names.index(param)]
            mins[param] = min(values)
            maxs[param] = max(values)
        # Create the log.param file
        for param in param_names:
            if param in NS_arguments['clustering_params']:
                mini, maxi = '%.6e'%mins[param], '%.6e'%maxs[param]
            else:
                mini, maxi = param_data[param][1], param_data[param][2]
            scaling = param_data[param][4]
            ptype   = param_data[param][5]
            line = pre+"'"+param+"'"+pos
            values = [MAPs[param], mini, maxi, sigmas[param], scaling, ptype]
            line += ' = [' + ', '.join(values) + ']\n'
            this_log_lines[param_lines[param]] = line
        # Write it!
        with open(os.path.join(mode_subfolder, 'log.param'), 'w') as log_file:
            log_file.writelines(this_log_lines)
