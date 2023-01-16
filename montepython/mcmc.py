"""
.. module:: mcmc
   :synopsis: Monte Carlo procedure
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

This module defines one key function, :func:`chain`, that handles the Markov
chain. So far, the code uses only one chain, as no parallelization is done.

The following routine is also defined in this module, which is called at
every step:

* :func:`get_new_position` returns a new point in the parameter space,
  depending on the proposal density.

The :func:`chain` in turn calls several helper routines, defined in
:mod:`sampler`. These are called just once:

* :func:`compute_lkl() <sampler.compute_lkl>` is called at every step in the Markov chain, returning
  the likelihood at the current point in the parameter space.
* :func:`get_covariance_matrix() <sampler.get_covariance_matrix>`
* :func:`read_args_from_chain() <sampler.read_args_from_chain>`
* :func:`read_args_from_bestfit() <sampler.read_args_from_bestfit>`
* :func:`accept_step() <sampler.accept_step>`

Their usage is described in :mod:`sampler`. On the contrary, the following
routines are called at every step:

The arguments of these functions will often contain **data** and/or **cosmo**.
They are both initialized instances of respectively :class:`data` and the
cosmological class. They will thus not be described for every function.
"""

from __future__ import print_function
import os
import sys
import math
import random as rd
import numpy as np
import warnings
import scipy.linalg as la
from pprint import pprint

import io_mp
import sampler

# Python 2.x - 3.x compatibility: Always use more efficient range function
try:
    xrange
except NameError:
    xrange = range


def get_new_position(data, eigv, U, k, Cholesky, Rotation):
    """
    Obtain a new position in the parameter space from the eigen values of the
    inverse covariance matrix, or from the Cholesky decomposition (original
    idea by Anthony Lewis, in `Efficient sampling of fast and slow
    cosmological parameters <http://arxiv.org/abs/1304.4473>`_ )

    The three different jumping options, decided when starting a run with the
    flag **-j**  are **global**, **sequential** and **fast** (by default) (see
    :mod:`parser_mp` for reference).

    .. warning::

        For running Planck data, the option **fast** is highly recommended, as
        it speeds up the convergence. Note that when using this option, the
        list of your likelihoods in your parameter file **must match** the
        ordering of your nuisance parameters (as always, they must come after
        the cosmological parameters, but they also must be ordered between
        likelihood, with, preferentially, the slowest likelihood to compute
        coming first).


    - **global**: varies all the parameters at the same time. Depending on the
      input covariance matrix, some degeneracy direction will be followed,
      otherwise every parameter will jump independently of each other.
    - **sequential**: varies every parameter sequentially. Works best when
      having no clue about the covariance matrix, or to understand which
      estimated sigma is wrong and slowing down the whole process.
    - **fast**: privileged method when running the Planck likelihood. Described
      in the aforementioned article, it separates slow (cosmological) and fast
      (nuisance) parameters.

    Parameters
    ----------
    eigv : numpy array
        Eigenvalues previously computed
    U : numpy_array
        Covariance matrix.
    k : int
        Number of points so far in the chain, is used to rotate through
        parameters
    Cholesky : numpy array
        Cholesky decomposition of the covariance matrix, and its inverse
    Rotation : numpy_array
        Not used yet

    """

    parameter_names = data.get_mcmc_parameters(['varying'])
    vector_new = np.zeros(len(parameter_names), 'float64')
    sigmas = np.zeros(len(parameter_names), 'float64')

    # Write the vector of last accepted points, or if it does not exist
    # (initialization routine), take the mean value
    vector = np.zeros(len(parameter_names), 'float64')
    try:
        for elem in parameter_names:
            vector[parameter_names.index(elem)] = \
                data.mcmc_parameters[elem]['last_accepted']
    except KeyError:
        for elem in parameter_names:
            vector[parameter_names.index(elem)] = \
                data.mcmc_parameters[elem]['initial'][0]

    # Initialize random seed
    rd.seed()

    # Choice here between sequential and global change of direction
    if data.jumping == 'global':
        for i in range(len(vector)):
            sigmas[i] = (math.sqrt(1/eigv[i]/len(vector))) * \
                rd.gauss(0, 1)*data.jumping_factor
    elif data.jumping == 'sequential':
        i = k % len(vector)
        sigmas[i] = (math.sqrt(1/eigv[i]))*rd.gauss(0, 1)*data.jumping_factor
    elif data.jumping == 'fast':
        #i = k % len(vector)
        j = k % len(data.over_sampling_indices)
        i = data.over_sampling_indices[j]
        ###############
        # method fast+global
        for index, elem in enumerate(data.block_parameters):
            # When the running index is below the maximum index of a block of
            # parameters, this block is varied, and **only this one** (note the
            # break at the end of the if clause, it is not a continue)
            if i < elem:
                if index == 0:
                    Range = elem
                    Previous = 0
                else:
                    Range = elem-data.block_parameters[index-1]
                    Previous = data.block_parameters[index-1]
                # All the varied parameters are given a random variation with a
                # sigma of 1. This will translate in a jump for all the
                # parameters (as long as the Cholesky matrix is non diagonal)
                for j in range(Range):
                    sigmas[j+Previous] = (math.sqrt(1./Range)) * \
                        rd.gauss(0, 1)*data.jumping_factor
                break
            else:
                continue
    else:
        print('\n\n Jumping method unknown (accepted : ')
        print('global, sequential, fast (default))')

    # Fill in the new vector
    if data.jumping in ['global', 'sequential']:
        vector_new = vector + np.dot(U, sigmas)
    else:
        vector_new = vector + np.dot(Cholesky, sigmas)

    # Check for boundaries problems
    flag = 0
    for i, elem in enumerate(parameter_names):
        value = data.mcmc_parameters[elem]['initial']
        if((str(value[1]) != str(-1) and value[1] is not None) and
                (vector_new[i] < value[1])):
            flag += 1  # if a boundary value is reached, increment
        elif((str(value[2]) != str(-1) and value[2] is not None) and
                vector_new[i] > value[2]):
            flag += 1  # same

    # At this point, if a boundary condition is not fullfilled, ie, if flag is
    # different from zero, return False
    if flag != 0:
        return False

    # Check for a slow step (only after the first time, so we put the test in a
    # try: statement: the first time, the exception KeyError will be raised)
    try:
        data.check_for_slow_step(vector_new)
    except KeyError:
        pass

    # If it is not the case, proceed with normal computation. The value of
    # new_vector is then put into the 'current' point in parameter space.
    for index, elem in enumerate(parameter_names):
        data.mcmc_parameters[elem]['current'] = vector_new[index]

    # Propagate the information towards the cosmo arguments
    data.update_cosmo_arguments()

    return True


######################
# MCMC CHAIN
######################
def chain(cosmo, data, command_line):
    """
    Run a Markov chain of fixed length with a Metropolis Hastings algorithm.

    Main function of this module, this is the actual Markov chain procedure.
    After having selected a starting point in parameter space defining the
    first **last accepted** one, it will, for a given amount of steps :

    + choose randomly a new point following the *proposal density*,
    + compute the cosmological *observables* through the cosmological module,
    + compute the value of the *likelihoods* of the desired experiments at this
      point,
    + *accept/reject* this point given its likelihood compared to the one of
      the last accepted one.

    Every time the code accepts :code:`data.write_step` number of points
    (quantity defined in the input parameter file), it will write the result to
    disk (flushing the buffer by forcing to exit the output file, and reopen it
    again.

    .. note::

        to use the code to set a fiducial file for certain fixed parameters,
        you can use two solutions. The first one is to put all input 1-sigma
        proposal density to zero (this method still works, but is not
        recommended anymore). The second one consist in using the flag "-f 0",
        to force a step of zero amplitude.

    """

    ## Initialisation
    loglike = 0

    # In case command_line.silent has been asked, outputs should only contain
    # data.out. Otherwise, it will also contain sys.stdout
    outputs = [data.out]
    if not command_line.silent:
        outputs.append(sys.stdout)

    use_mpi = False
    # check for MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # suppress duplicate output from slaves
        if rank:
            command_line.quiet = True
        use_mpi = True
    except ImportError:
        # set all chains to master if no MPI
        rank = 0

    # Initialise master and slave chains for superupdate.
    # Workaround in order to have one master chain and several slave chains even when
    # communication fails between MPI chains. It could malfunction on some hardware.
    # TODO: Would like to merge with MPI initialization above and make robust and logical
    # TODO: Or if keeping current scheme, store value and delete jumping_factor.txt
    # TODO: automatically if --parallel-chains is enabled
    if command_line.superupdate and data.jumping_factor:
        try:
            jump_file = open(command_line.folder + '/jumping_factor.txt','r')
            #if command_line.restart is None:
            if not use_mpi and command_line.parallel_chains:
                rank = 1
                warnings.warn('MPI not in use, flag --parallel-chains enabled, '
                              'superupdate enabled, and a jumping_factor.txt file detected. '
                              'If relaunching in the same folder or restarting a run this '
                              'will cause all chains to be assigned as slaves. In this case '
                              'instead note the value in jumping_factor.txt, delete the '
                              'file, and pass the value with flag -f <value>. This warning '
                              'may then appear again, but you can safely disregard it.')
            else:
                # For restart runs we want to save the input jumping factor
                # as starting jumping factor, but continue from the jumping
                # factor stored in the file.
                starting_jumping_factor = data.jumping_factor
                # This will load the value irrespective of whether it starts
                # with # (i.e. the jumping factor adaptation was started) or not.
                jump_value = jump_file.read().replace('# ','')
                data.jumping_factor = float(jump_value)
            jump_file.close()
            print('rank = ',rank)
        except:
            jump_file = open(command_line.folder + '/jumping_factor.txt','w')
            jump_file.write(str(data.jumping_factor))
            jump_file.close()
            rank = 0
            print('rank = ',rank)
            starting_jumping_factor = data.jumping_factor

    # Recover the covariance matrix according to the input, if the varying set
    # of parameters is non-zero
    if (data.get_mcmc_parameters(['varying']) != []):

        # Read input covariance matrix
        sigma_eig, U, C = sampler.get_covariance_matrix(cosmo, data, command_line)

        # if we want to compute the starting point by minimising lnL (instead of taking it from input file or bestfit file)
        minimum = 0
        if command_line.minimize:
            minimum, min_chi2 = sampler.get_minimum(cosmo, data, command_line, C)

            parameter_names = data.get_mcmc_parameters(['last_accepted'])
            for index,elem in parameter_names:
                data.mcmc_parameters[elem]['last_accepted'] = minimum[index]

            #FK: write out the results of the minimzer:
            labels = data.get_mcmc_parameters(['varying'])
            fname = os.path.join(command_line.folder, 'results.minimized')
            with open(fname, 'w') as f:
                f.write('# minimized \chi^2 = {:} \n'.format(min_chi2))
                f.write('# %s\n' % ', '.join(['%16s' % label for label in labels]))
                for idx in xrange(len(labels)):
                    bf_value = minimum[idx]
                    if bf_value > 0:
                        f.write(' %.6e\t' % bf_value)
                    else:
                        f.write('%.6e\t' % bf_value)
                f.write('\n')
            print('Results of minimizer saved to: \n', fname)

        # if we want to compute Fisher matrix and then stop
        if command_line.fisher:
            sampler.get_fisher_matrix(cosmo, data, command_line, C, minimum)
            return

        # warning if no jumps are requested
        if data.jumping_factor == 0:
            warnings.warn(
                "The jumping factor has been set to 0. The above covariance " +
                "matrix will not be used.")

    # In case of a fiducial run (all parameters fixed), simply run once and
    # print out the likelihood. This should not be used any more (one has to
    # modify the log.param, which is never a good idea. Instead, force the code
    # to use a jumping factor of 0 with the option "-f 0".
    else:
        warnings.warn(
            "You are running with no varying parameters... I will compute " +
            "only one point and exit")
        data.update_cosmo_arguments()  # this fills in the fixed parameters
        loglike = sampler.compute_lkl(cosmo, data)
        io_mp.print_vector(outputs, 1, loglike, data)
        return 1, loglike

    # In the fast-slow method, one need the Cholesky decomposition of the
    # covariance matrix. Return the Cholesky decomposition as a lower
    # triangular matrix
    Cholesky = None
    Rotation = None
    if command_line.jumping == 'fast':
        Cholesky = la.cholesky(C).T
        Rotation = np.identity(len(sigma_eig))

    # define path and covmat
    input_covmat = command_line.cov
    base = os.path.basename(command_line.folder)
    # the previous line fails when "folder" is a string ending with a slash. This issue is cured by the next lines:
    if base == '':
        base = os.path.basename(command_line.folder[:-1])
    command_line.cov = os.path.join(
        command_line.folder, base+'.covmat')

    # Fast Parameter Multiplier (fpm) for adjusting update and superupdate numbers.
    # This is equal to N_slow + f_fast N_fast, where N_slow is the number of slow
    # parameters, f_fast is the over sampling number for each fast block and f_fast
    # is the number of parameters in each fast block.
    for i in range(len(data.block_parameters)):
        if i == 0:
            fpm = data.over_sampling[i]*data.block_parameters[i]
        else:
            fpm += data.over_sampling[i]*(data.block_parameters[i] - data.block_parameters[i-1])

    # If the update mode was selected, the previous (or original) matrix should be stored
    if command_line.update:
        if not rank and not command_line.silent:
            print('Update routine is enabled with value %d (recommended: 50)' % command_line.update)
            print('This number is rescaled by cycle length %d (N_slow + f_fast * N_fast) to %d' % (fpm,fpm*command_line.update))
        # Rescale update number by cycle length N_slow + f_fast * N_fast to account for fast parameters
        command_line.update *= fpm
        previous = (sigma_eig, U, C, Cholesky)

    # Initialise adaptive
    if command_line.adaptive:
        if not command_line.silent:
            print('Adaptive routine is enabled with value %d (recommended: 10*dimension)' % command_line.adaptive)
            print('and adaptive_ts = %d (recommended: 100*dimension)' % command_line.adaptive_ts)
            print('Please note: current implementation not suitable for multiple chains')
        if rank > 0:
            raise io_mp.ConfigurationError('Adaptive routine not compatible with MPI')
        if command_line.update:
            warnings.warn('Adaptive routine not compatible with update, overwriting input update value')
        if command_line.superupdate:
            warnings.warn('Adaptive routine not compatible with superupdate, deactivating superupdate')
            command_line.superupdate = 0
        # Define needed parameters
        parameter_names = data.get_mcmc_parameters(['varying'])
        mean = np.zeros(len(parameter_names))
        last_accepted = np.zeros(len(parameter_names),'float64')
        ar = np.zeros(100)
        if command_line.cov == None:
            # If no input covmat was given, the starting jumping factor
            # should be very small until a covmat is obtained and the
            # original start jumping factor should be saved
            start_jumping_factor = command_line.jumping_factor
            data.jumping_factor = command_line.jumping_factor/100.
            # Analyze module will be forced to compute one covmat,
            # after which update flag will be set to False.
            command_line.update = command_line.adaptive
        else:
            # If an input covmat was provided, take mean values from param file
            # Question: is it better to always do this, rather than setting mean
            # to last accepted after the initial update run?
            for elem in parameter_names:
                mean[parameter_names.index(elem)] = data.mcmc_parameters[elem]['initial'][0]

    # Initialize superupdate
    if command_line.superupdate:
        if not rank and not command_line.silent:
            print('Superupdate routine is enabled with value %d (recommended: 20)' % command_line.superupdate)
            if command_line.superupdate < 20:
                warnings.warn('Superupdate value lower than the recommended value. This '
                              'may increase the risk of poorly converged acceptance rate')
            print('This number is rescaled by cycle length %d (N_slow + f_fast * N_fast) to %d' % (fpm,fpm*command_line.superupdate))
        # Rescale superupdate number by cycle length N_slow + f_fast * N_fast to account for fast parameters
        command_line.superupdate *= fpm
        # Define needed parameters
        parameter_names = data.get_mcmc_parameters(['varying'])
        updated_steps = 0
        stop_c = False
        jumping_factor_rescale = 0
        if command_line.restart:
            try:
                jump_file = open(command_line.cov,'r')
                jumping_factor_rescale = 1
            except:
                jumping_factor_rescale = 0
        c_array = np.zeros(command_line.superupdate) # Allows computation of mean of jumping factor
        R_minus_one = np.array([100.,100.]) # 100 to make sure max(R-1) value is high if computation failed
        # Local acceptance rate of last SU*(N_slow + f_fast * N_fast) steps
        ar = np.zeros(command_line.superupdate)
        # Store acceptance rate of last 5*SU*(N_slow + f_fast * N_fast) steps
        backup_ar = np.zeros(5*command_line.superupdate)
        # Make sure update is enabled
        if command_line.update == 0:
            if not rank and not command_line.silent:
                print('Update routine required by superupdate. Setting --update 50')
                print('This number is then rescaled by cycle length: %d (N_slow + f_fast * N_fast)' % fpm)
            command_line.update = 50 * fpm
            previous = (sigma_eig, U, C, Cholesky)

    # If restart wanted, pick initial value for arguments
    if command_line.restart is not None:
        sampler.read_args_from_chain(data, command_line.restart)

    # If restart from best fit file, read first point (overwrite settings of
    # read_args_from_chain)
    if command_line.bf is not None and not command_line.minimize:
        sampler.read_args_from_bestfit(data, command_line.bf)

    # Pick a position (from last accepted point if restart, from the mean value
    # else), with a 100 tries.
    for i in range(100):
        if get_new_position(data, sigma_eig, U, i,
                            Cholesky, Rotation) is True:
            break
        if i == 99:
            raise io_mp.ConfigurationError(
                "You should probably check your prior boundaries... because " +
                "no valid starting position was found after 100 tries")

    # Compute the starting Likelihood
    loglike = sampler.compute_lkl(cosmo, data)

    # Choose this step as the last accepted value
    # (accept_step), and modify accordingly the max_loglike
    sampler.accept_step(data)
    max_loglike = loglike

    # If the jumping factor is 0, the likelihood associated with this point is
    # displayed, and the code exits.
    if data.jumping_factor == 0:
        io_mp.print_vector(outputs, 1, loglike, data)
        return 1, loglike

    acc, rej = 0.0, 0.0  # acceptance and rejection number count
    N = 1   # number of time the system stayed in the current position

    # Print on screen the computed parameters
    if not command_line.silent and not command_line.quiet:
        io_mp.print_parameters(sys.stdout, data)

    # Suppress non-informative output after initializing
    command_line.quiet = True

    k = 1
    # Main loop, that goes on while the maximum number of failure is not
    # reached, and while the expected amount of steps (N) is not taken.
    while k <= command_line.N:
        # If the number of steps reaches the number set in the adaptive method plus one,
        # then the proposal distribution should be gradually adapted.
        # If the number of steps also exceeds the number set in adaptive_ts,
        # the jumping factor should be gradually adapted.
        if command_line.adaptive and k>command_line.adaptive+1:
            # Start of adaptive routine
            # By B. Schroer and T. Brinckmann
            # Modified version of the method outlined in the PhD thesis of Marta Spinelli

            # Store last accepted step
            for elem in parameter_names:
                last_accepted[parameter_names.index(elem)] = data.mcmc_parameters[elem]['last_accepted']
            # Recursion formula for mean and covmat (and jumping factor after ts steps)
            # mean(k) = mean(k-1) + (last_accepted - mean(k-1))/k
            mean += 1./k*(last_accepted-mean)
            # C(k) = C(k-1) + [(last_accepted - mean(k))^T * (last_accepted - mean(k)) - C(k-1)]/k
            C +=1./k*(np.dot(np.transpose(np.asmatrix(last_accepted-mean)),np.asmatrix(last_accepted-mean))-C)
            sigma_eig, U = np.linalg.eig(np.linalg.inv(C))
            if command_line.jumping == 'fast':
                Cholesky = la.cholesky(C).T
            if k>command_line.adaptive_ts:
                # c = j^2/d
                c = data.jumping_factor**2/len(parameter_names)
                # c(k) = c(k-1) + [acceptance_rate(last 100 steps) - 0.25]/k
                c +=(np.mean(ar)-0.25)/k
                data.jumping_factor = np.sqrt(len(parameter_names)*c)

            # Save the covariance matrix and the jumping factor in a file
            # For a possible MPI implementation
            #if not (k-command_line.adaptive) % 5:
            #    io_mp.write_covariance_matrix(C,parameter_names,str(command_line.cov))
            #    jump_file = open(command_line.folder + '/jumping_factor.txt','w')
            #    jump_file.write(str(data.jumping_factor))
            #    jump_file.close()
            # End of adaptive routine

        # If the number of steps reaches the number set in the update method,
        # then the proposal distribution should be adapted.
        if command_line.update:
            # Start of update routine
            # By M. Ballardini and T. Brinckmann
            # Also used by superupdate and adaptive

            # master chain behavior
            if not rank:
                # Add the folder to the list of files to analyze, and switch on the
                # options for computing only the covmat
                from parser_mp import parse
                info_command_line = parse(
                    'info %s --minimal --noplot --keep-fraction 0.5 --keep-non-markovian --want-covmat' % command_line.folder)
                info_command_line.update = command_line.update

                if command_line.adaptive:
                    # Keep all points for covmat guess in adaptive
                    info_command_line = parse('info %s --minimal --noplot --keep-non-markovian --want-covmat' % command_line.folder)
                    # Tell the analysis to update the covmat after t0 steps if it is adaptive
                    info_command_line.adaptive = command_line.adaptive
                    # Only compute covmat if no input covmat was provided
                    if input_covmat != None:
                        info_command_line.want_covmat = False

                # This is in order to allow for more frequent R-1 computation with superupdate
                compute_R_minus_one = False
                if command_line.superupdate:
                    if not (k+10) % command_line.superupdate:
                        compute_R_minus_one = True
                # the +10 below is here to ensure that the first master update will take place before the first slave updates,
                # but this is a detail, the code is robust against situations where updating is not possible, so +10 could be omitted
                if (not (k+10) % command_line.update or compute_R_minus_one) and k > 10:
                    # Try to launch an analyze (computing a new covmat if successful)
                    try:
                        if not (k+10) % command_line.update:
                            from analyze import analyze
                            R_minus_one = analyze(info_command_line)
                        elif command_line.superupdate:
                            # Compute (only, i.e. no covmat) R-1 more often when using superupdate
                            info_command_line = parse(
                                'info %s --minimal --noplot --keep-fraction 0.5 --keep-non-markovian' % command_line.folder)
                            info_command_line.update = command_line.update
                            R_minus_one = analyze(info_command_line)
                    except:
                        if not command_line.silent:
                            print('Step ',k,' chain ', rank,': Failed to calculate covariance matrix')

                if command_line.superupdate:
                    # Start of superupdate routine
                    # By B. Schroer and T. Brinckmann

                    c_array[(k-1)%(command_line.superupdate)] = data.jumping_factor

                    # If acceptance rate deviates too much from the target acceptance
                    # rate we want to resume adapting the jumping factor
                    # T. Brinckmann 02/2019: use mean a.r. over the last 5*len(ar) steps
                    # instead or the over last len(ar), which is more stable
                    if abs(np.mean(backup_ar) - command_line.superupdate_ar) > 5.*command_line.superupdate_ar_tol:
                        stop_c = False

                    # Start adapting the jumping factor after command_line.superupdate steps if R-1 < 10
                    # The lower R-1 criterium is an arbitrary choice to keep from updating when the R-1
                    # calculation fails (i.e. returns only zeros).
                    if (k > updated_steps + command_line.superupdate) and 0.01 < (max(R_minus_one) < 10.) and not stop_c:
                        c = data.jumping_factor**2/len(parameter_names)
                        # To avoid getting trapped in local minima, the jumping factor should
                        # not go below 0.1 (arbitrary) times the starting jumping factor.
                        if (c + (np.mean(ar) - command_line.superupdate_ar)/(k - updated_steps)) > (0.1*starting_jumping_factor)**2./len(parameter_names) or ((np.mean(ar) - command_line.superupdate_ar)/(k - updated_steps) > 0):
                            c += (np.mean(ar) - command_line.superupdate_ar)/(k - updated_steps)
                            data.jumping_factor = np.sqrt(len(parameter_names) * c)

                        if not (k-1) % 5:
                            # Check if the jumping factor adaptation should stop.
                            # An acceptance rate of 25% balances the wish for more accepted
                            # points, while ensuring the parameter space is properly sampled.
                            # The convergence criterium is by default (26+/-1)%, so the adaptation
                            # will stop when the code reaches an acceptance rate of at least 25%.
                            # T. Brinckmann 02/2019: use mean a.r. over the last 5*len(ar) steps
                            # instead or the over last len(ar), which is more stable
                            if (max(R_minus_one) < 0.4) and (abs(np.mean(backup_ar) - command_line.superupdate_ar) < command_line.superupdate_ar_tol) and (abs(np.mean(c_array)/c_array[(k-1) % (command_line.superupdate)] - 1) < 0.01):
                                stop_c = True
                                data.out.write('# After %d accepted steps: stop adapting the jumping factor at a value of %f with a local acceptance rate %f \n' % (int(acc),data.jumping_factor,np.mean(backup_ar)))
                                if not command_line.silent:
                                    print('After %d accepted steps: stop adapting the jumping factor at a value of %f with a local acceptance rate of %f \n' % (int(acc), data.jumping_factor,np.mean(backup_ar)))
                                jump_file = open(command_line.folder + '/jumping_factor.txt','w')
                                jump_file.write('# '+str(data.jumping_factor))
                                jump_file.close()
                            else:
                                jump_file = open(command_line.folder + '/jumping_factor.txt','w')
                                jump_file.write(str(data.jumping_factor))
                                jump_file.close()

                    # Write the evolution of the jumping factor to a file
                    if not k % (command_line.superupdate):
                        jump_file = open(command_line.folder + '/jumping_factors.txt','a')
                        for i in xrange(command_line.superupdate):
                            jump_file.write(str(c_array[i])+'\n')
                        jump_file.close()
                    # End of main part of superupdate routine

                if not (k-1) % (command_line.update//3):
                    try:
                        # Read the covmat
                        sigma_eig, U, C = sampler.get_covariance_matrix(
                            cosmo, data, command_line)
                        if command_line.jumping == 'fast':
                            Cholesky = la.cholesky(C).T
                        # Test here whether the covariance matrix has really changed
                        # We should in principle test all terms, but testing the first one should suffice
                        if not C[0,0] == previous[2][0,0]:
                            if k == 1:
                                if not command_line.silent:
                                    if not input_covmat == None:
                                        warnings.warn(
                                            'Appending to an existing folder: using %s instead of %s. '
                                            'If new input covmat is desired, please delete previous covmat.'
                                            % (command_line.cov, input_covmat))
                                    else:
                                        warnings.warn(
                                            'Appending to an existing folder: using %s. '
                                            'If no starting covmat is desired, please delete previous covmat.'
                                            % command_line.cov)
                            else:
                                # Start of second part of superupdate routine
                                if command_line.superupdate:
                                    # Adaptation of jumping factor should start again after the covmat is updated
                                    # Save the step number after it updated for superupdate and start adaption of c again
                                    updated_steps = k
                                    stop_c = False
                                    cov_det = np.linalg.det(C)
                                    prev_cov_det = np.linalg.det(previous[2])
                                    # Rescale jumping factor in order to keep the magnitude of the jumps the same.
                                    # Skip this update the first time the covmat is updated in order to prevent
                                    # problems due to a poor initial covmat. Rescale the jumping factor after the
                                    # first calculated covmat to the expected optimal one of 2.4.
                                    if jumping_factor_rescale:
                                        new_jumping_factor = data.jumping_factor * (prev_cov_det/cov_det)**(1./(2 * len(parameter_names)))
                                        data.out.write('# After %d accepted steps: rescaled jumping factor from %f to %f, due to updated covariance matrix \n' % (int(acc), data.jumping_factor, new_jumping_factor))
                                        if not command_line.silent:
                                            print('After %d accepted steps: rescaled jumping factor from %f to %f, due to updated covariance matrix \n' % (int(acc), data.jumping_factor, new_jumping_factor))
                                        data.jumping_factor = new_jumping_factor
                                    else:
                                        data.jumping_factor = starting_jumping_factor
                                    jumping_factor_rescale += 1
                                # End of second part of superupdate routine

                                # Write to chains file when the covmat was updated
                                data.out.write('# After %d accepted steps: update proposal with max(R-1) = %f and jumping factor = %f \n' % (int(acc), max(R_minus_one), data.jumping_factor))
                                if not command_line.silent:
                                    print('After %d accepted steps: update proposal with max(R-1) = %f and jumping factor = %f \n' % (int(acc), max(R_minus_one), data.jumping_factor))
                                try:
                                    if stop_after_update:
                                        k = command_line.N
                                        print('Covariance matrix updated - stopping run')
                                except:
                                    pass

                            previous = (sigma_eig, U, C, Cholesky)
                    except:
                        pass

                    command_line.quiet = True

                    # Start of second part of adaptive routine
                    # Stop updating the covmat after t0 steps in adaptive
                    if command_line.adaptive and k > 1:
                        command_line.update = 0
                        data.jumping_factor = start_jumping_factor
                        # Test if there are still enough steps left before the adaption of the jumping factor starts
                        if k > 0.5*command_line.adaptive_ts:
                            command_line.adaptive_ts += k
                        # Set the mean for the recursion formula to the last accepted point
                        for elem in parameter_names:
                            mean[parameter_names.index(elem)] = data.mcmc_parameters[elem]['last_accepted']
                    # End of second part of adaptive routine

            # slave chain behavior
            else:
                # Start of slave superupdate routine
                if command_line.superupdate:
                    # If acceptance rate deviates too much from the target acceptance
                    # rate we want to resume adapting the jumping factor. This line
                    # will force the slave chains to check if the jumping factor
                    # has been updated
                    if abs(np.mean(backup_ar) - command_line.superupdate_ar) > 5.*command_line.superupdate_ar_tol:
                        stop_c = False

                    # Update the jumping factor every 5 steps in superupdate
                    if not k % 5 and k > command_line.superupdate and command_line.superupdate and (not stop_c or (stop_c and k % command_line.update)):
                        try:
                            jump_file = open(command_line.folder + '/jumping_factor.txt','r')
                            # If there is a # in the file, the master has stopped adapting c
                            for line in jump_file:
                                if line.find('#') == -1:
                                    jump_file.seek(0)
                                    jump_value = jump_file.read()
                                    data.jumping_factor = float(jump_value)
                                else:
                                    jump_file.seek(0)
                                    jump_value = jump_file.read().replace('# ','')
                                    #if not stop_c or (stop_c and not float(jump_value) == data.jumping_factor):
                                    if not float(jump_value) == data.jumping_factor:
                                        data.jumping_factor = float(jump_value)
                                        stop_c = True
                                        data.out.write('# After %d accepted steps: stop adapting the jumping factor at a value of %f with a local acceptance rate %f \n' % (int(acc),data.jumping_factor,np.mean(backup_ar)))
                                        if not command_line.silent:
                                            print('After %d accepted steps: stop adapting the jumping factor at a value of %f with a local acceptance rate of %f \n' % (int(acc), data.jumping_factor,np.mean(backup_ar)))
                            jump_file.close()
                        except:
                            if not command_line.silent:
                                print('Reading jumping_factor file failed')
                            pass
                # End of slave superupdate routine

                # Start of slave update routine
                if not (k-1) % (command_line.update//10):
                    try:
                        sigma_eig, U, C = sampler.get_covariance_matrix(
                            cosmo, data, command_line)
                        if command_line.jumping == 'fast':
                            Cholesky = la.cholesky(C).T
                        # Test here whether the covariance matrix has really changed
                        # We should in principle test all terms, but testing the first one should suffice
                        if not C[0,0] == previous[2][0,0] and not k == 1:
                            if command_line.superupdate:
                                # If the covmat was updated, the master has resumed adapting c
                                stop_c = False
                            data.out.write('# After %d accepted steps: update proposal \n' % int(acc))
                            if not command_line.silent:
                                print('After %d accepted steps: update proposal \n' % int(acc))
                            try:
                                if stop_after_update:
                                    k = command_line.N
                                    print('Covariance matrix updated - stopping run')
                            except:
                                pass
                        previous = (sigma_eig, U, C, Cholesky)

                    except:
                        pass
                # End of slave update routine
            # End of update routine

        # Pick a new position ('current' flag in mcmc_parameters), and compute
        # its likelihood. If get_new_position returns True, it means it did not
        # encounter any boundary problem. Otherwise, just increase the
        # multiplicity of the point and start the loop again
        if get_new_position(
                data, sigma_eig, U, k, Cholesky, Rotation) is True:
            newloglike = sampler.compute_lkl(cosmo, data)
        else:  # reject step
            rej += 1
            if command_line.superupdate:
                ar[k%len(ar)] = 0 # Local acceptance rate of last SU*(N_slow + f_fast * N_fast) steps
            elif command_line.adaptive:
                ar[k%len(ar)] = 0 # Local acceptance rate of last 100 steps
            N += 1
            k += 1
            continue

        # Harmless trick to avoid exponentiating large numbers. This decides
        # whether or not the system should move.
        if (newloglike != data.boundary_loglike):
            if (newloglike >= loglike):
                alpha = 1.
            else:
                alpha = np.exp(newloglike-loglike)
        else:
            alpha = -1

        if ((alpha == 1.) or (rd.uniform(0, 1) < alpha)):  # accept step

            # Print out the last accepted step (WARNING: this is NOT the one we
            # just computed ('current' flag), but really the previous one.)
            # with its proper multiplicity (number of times the system stayed
            # there).
            io_mp.print_vector(outputs, N, loglike, data)

            # Report the 'current' point to the 'last_accepted'
            sampler.accept_step(data)
            loglike = newloglike
            if loglike > max_loglike:
                max_loglike = loglike
            acc += 1.0
            N = 1  # Reset the multiplicity
            if command_line.superupdate:
                ar[k%len(ar)] = 1 # Local acceptance rate of last SU*(N_slow + f_fast * N_fast) steps
            elif command_line.adaptive:
                ar[k%len(ar)] = 1 # Local acceptance rate of last 100 steps
        else:  # reject step
            rej += 1.0
            N += 1  # Increase multiplicity of last accepted point
            if command_line.superupdate:
                ar[k%len(ar)] = 0 # Local acceptance rate of last SU*(N_slow + f_fast * N_fast) steps
            elif command_line.adaptive:
                ar[k%len(ar)] = 0 # Local acceptance rate of last 100 steps

        # Store a.r. for last 5 x SU*(N_slow + f_fast * N_fast) steps
        if command_line.superupdate:
            backup_ar[k%len(backup_ar)] = ar[k%len(ar)]

        # Regularly (option to set in parameter file), close and reopen the
        # buffer to force to write on file.
        try:
            data.write_step
        except:
            # Set default value to 5 if not specified by user
            data.write_step = 5
        if acc % data.write_step == 0:
            io_mp.refresh_file(data)
            # Update the outputs list
            outputs[0] = data.out
        k += 1  # One iteration done
    # END OF WHILE LOOP

    # If at this moment, the multiplicity is higher than 1, it means the
    # current point is not yet accepted, but it also mean that we did not print
    # out the last_accepted one yet. So we do.
    if N > 1:
        io_mp.print_vector(outputs, N-1, loglike, data)

    # Print out some information on the finished chain
    rate = acc / (acc + rej)
    sys.stdout.write('\n#  {0} steps done, acceptance rate: {1}\n'.
                     format(command_line.N, rate))

    # In case the acceptance rate is too low, or too high, print a warning
    if rate < 0.05:
        warnings.warn("The acceptance rate is below 0.05. You might want to "
                      "set the jumping factor to a lower value than the "
                      "default (2.4), with the option `-f 1.5` for instance.")
    elif rate > 0.6:
        warnings.warn("The acceptance rate is above 0.6, which means you might"
                      " have difficulties exploring the entire parameter space"
                      ". Try analysing these chains, and use the output "
                      "covariance matrix to decrease the acceptance rate to a "
                      "value between 0.2 and 0.4 (roughly).")
    # For a restart, erase the starting point to keep only the new, longer
    # chain.
    if command_line.restart is not None:
        os.remove(command_line.restart)
        sys.stdout.write('    deleting starting point of the chain {0}\n'.
                         format(command_line.restart))

    return
