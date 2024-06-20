"""
.. module:: analyze
   :synopsis: Extract data from chains and produce plots

.. moduleauthor:: Karim Benabed <benabed@iap.fr>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Collection of functions needed to analyze the Markov chains.

This module defines as well a class :class:`Information`, that stores useful
quantities, and shortens the argument passing between the functions.

.. note::
    Some of the methods used in this module are directly adapted from the
    `CosmoPmc <http://www.cosmopmc.info>`_ code from Kilbinger et. al.

"""
from __future__ import print_function
import os
import math
import numpy as np
from itertools import count
# The root plotting module, to change options like font sizes, etc...
import matplotlib
# The following line suppresses the need for an X server
matplotlib.use("Agg")
# Module for handling display
import matplotlib.pyplot as plt
# Module to handle warnings from matplotlib
import warnings
import importlib
import io_mp
try:
    from itertools import ifilterfalse as py_filterfalse
except ImportError:
    from itertools import filterfalse as py_filterfalse
try:
    from itertools import ifilter as py_filter
except ImportError:
    from builtins import filter as py_filter

import scipy.ndimage
import scipy.special
import numpy.linalg as la
import sys
from io_mp import dictitems,dictvalues,dictkeys

# Defined to remove the burnin for all the points that were produced before the
# first time where -log-likelihood <= min-minus-log-likelihood+LOG_LKL_CUTOFF
#### Change the following line for MultiNest to 1e300
#### as we do not want to remove burn-in
LOG_LKL_CUTOFF = 3
# LOG_LKL_CUTOFF = 1e300

NUM_COLORS = 6

# Python 2.x - 3.x compatibility: Always use more efficient range function
try:
    xrange
except NameError:
    xrange = range


def analyze(command_line):
    """
    Main function, does the entire analysis.

    It calls in turn all the other routines from this module. To limit the
    arguments of each function to a reasonnable size, a :class:`Information`
    instance is used. This instance is initialized in this function, then
    appended by the other routines.

    """
    # Check if the scipy module has the interpolate method correctly
    # installed (should be the case on every linux distribution with
    # standard numpy)
    try:
        from scipy.interpolate import interp1d
        Information.has_interpolate_module = True
    except ImportError:
        Information.has_interpolate_module = False
        warnings.warn(
            'No cubic interpolation done (no interpolate method found ' +
            'in scipy), only linear')

    # Determine how many different folders are asked through the 'info'
    # command, and create as many Information instances
    files = separate_files(command_line.files)

    # Create an instance of the Information class for each subgroup found in
    # the previous function. They will each hold all relevant information, and
    # be used as a compact way of exchanging information between functions
    information_instances = []
    for item in files:
        info = Information(command_line)
        information_instances.append(info)

        # Prepare the files, according to the case, load the log.param, and
        # prepare the output (plots folder, .covmat, .info and .log files).
        # After this step, info.files will contain all chains.
        status = prepare(item, info)
        # If the preparation step generated new files (for instance,
        # translating from NS or CH to Markov Chains) this routine should stop
        # now.
        if not status:
            return

        # Compute the mean, maximum of likelihood, 1-sigma variance for this
        # main folder. This will create the info.chain object, which contains
        # all the points computed stacked in one big array.
        convergence(info)

        # check if analyze() is called directly by the user, or by the mcmc loop during an updating phase
        try:
            # command_line.update is defined when called by the mcmc loop
            command_line.update
        except:
            # in case it was not defined (i.e. when analyze() is called directly by user), set it to False
            command_line.update = 0
        # check if analyze() is called directly by the user, or by the mcmc loop at the start of an adaptive run
        try:
            # command_line.adaptive is defined when called by the mcmc loop
            command_line.adaptive
        except:
            # in case it was not defined (i.e. when analyze() is called directly by user), set it to False
            command_line.adaptive = 0

        # compute covariance matrix, except when we are in update mode and convergence is too bad or good enough
        # or if we are in adaptive mode and only want a first guess for the covmat
        if command_line.update and (np.amax(info.R) > 3. or np.amax(info.R) < 0.4 or np.isnan(np.sum(info.R))) and not command_line.adaptive:
            print('--> Not computing covariance matrix')
        else:
            try:
                if command_line.want_covmat:
                    print('--> Computing covariance matrix')
                    info.covar = compute_covariance_matrix(info)
                    # Writing it out in name_of_folder.covmat
                    io_mp.write_covariance_matrix(
                        info.covar, info.backup_names, info.cov_path)
            except:
                print('--> Computing covariance matrix failed')
                pass

        # Store an array, sorted_indices, containing the list of indices
        # corresponding to the line with the highest likelihood as the first
        # element, and then as decreasing likelihood
        info.sorted_indices = info.chain[:, 1].argsort(0)

        # Writing the best-fit model in name_of_folder.bestfit
        bestfit_line = [elem*info.scales[i, i] for i, elem in
                        enumerate(info.chain[info.sorted_indices[0], 2:])]
        io_mp.write_bestfit_file(bestfit_line, info.backup_names,
                                 info.best_fit_path)

    # Overwrite center of Fisher matrix from log.param with the bestfit
    # from the last set of chains provided
    # DEBUG: This doesn't plot the first parameter (omega_b), possibly
    # because it's re-scaled (since it's the only one that is rescaled
    # and the rest are plotted)?
    if command_line.center_fisher:
        for index, elem in enumerate(info.ref_names):
            info.centers[index] = bestfit_line[index]/info.scales[index, index]

    if not command_line.minimal:
        # Computing 1,2 and 3-sigma errors, and plot. This will create the
        # triangle and 1d plot by default.
        compute_posterior(information_instances)

        print('--> Writing .info and .tex files')
        for info in information_instances:
            info.write_information_files()

    # when called by MCMC in update mode, return R values so that they can be written for information in the chains
    if command_line.update:
        return info.R

def prepare(files, info):
    """
    Scan the whole input folder, and include all chains in it.

    Since you can decide to analyze some file(s), or a complete folder, this
    function first needs to separate between the two cases.

    .. warning::
        If someday you change the way the chains are named, remember to change
        here too, because this routine assumes the chains have a double
        underscore in their names.

    .. note::
        Only files ending with .txt will be selected, to keep compatibility
        with CosmoMC format

    .. note::
        New in version 2.0.0: if you ask to analyze a MultiNest
        sub-folder (i.e. something that ends in `NS` with capital letters), the
        analyze module will translate the output from MultiNest to
        standard chains for Monte Python, and stops. You can then run the
        `-- info` flag on the whole folder. **This procedure is not necessary
        if the run was complete, but only if the MultiNest run was killed
        before completion**.

    Parameters
    ----------
    files : list
        list of potentially only one element, containing the files to analyze.
        This can be only one file, or the encompassing folder, files
    info : Information instance
        Used to store the result

    """
    # First test if the folder is a MultiNest, PolyChord or CosmoHammer folder.
    # If so, call the module's own routine through the clean conversion
    # function, which will translate the output of this other sampling into
    # MCMC chains that can then be analyzed.
    modules = ['MultiNest', 'PolyChord', 'cosmo_hammer']
    tags = ['NS', 'PC', 'CH']
    for module_name, tag in zip(modules, tags):
        action_done = clean_conversion(module_name, tag, files[0])
        if action_done:
            return False

    # If the input command was an entire folder, then grab everything in it.
    # Too small files (below 600 octets) and subfolders are automatically
    # removed.
    folder, files, basename = recover_folder_and_files(files)

    info.files = files
    info.folder = folder
    info.basename = basename

    # Check if the log.param file exists
    parameter_file_path = os.path.join(folder, 'log.param')
    if os.path.isfile(parameter_file_path):
        if os.path.getsize(parameter_file_path) == 0:
            raise io_mp.AnalyzeError(
                "The log param file %s " % os.path.join(folder, 'log.param') +
                "seems empty")
    else:
        raise io_mp.AnalyzeError(
            "The log param file %s " % os.path.join(folder, 'log.param') +
            "is missing in the analyzed folder?")

    # If the folder has no subdirectory, then go for a simple infoname,
    # otherwise, call it with the last name
    basename = (os.path.basename(folder) if os.path.basename(folder) != '.'
                else os.path.basename(os.path.abspath(
                    os.path.join(folder, '..'))))

    info.v_info_path = os.path.join(folder, basename+'.v_info')
    info.h_info_path = os.path.join(folder, basename+'.h_info')
    info.tex_path = os.path.join(folder, basename+'.tex')
    info.cov_path = os.path.join(folder, basename+'.covmat')
    info.log_path = os.path.join(folder, basename+'.log')
    info.best_fit_path = os.path.join(folder, basename+'.bestfit')
    info.param_path = parameter_file_path

    return True


def convergence(info):
    """
    Compute convergence for the desired chains, using Gelman-Rubin diagnostic

    Chains have been stored in the info instance of :class:`Information`. Note
    that the G-R diagnostic can be computed for a single chain, albeit it will
    most probably give absurd results. To do so, it separates the chain into
    three subchains.

    """
    # Recovering parameter names and scales, creating tex names,
    extract_parameter_names(info)

    # Circle through all files to find the global maximum of likelihood
    #print('--> Finding global maximum of likelihood')
    find_maximum_of_likelihood(info)

    # Restarting the circling through files, this time removing the burnin,
    # given the maximum of likelihood previously found and the global variable
    # LOG_LKL_CUTOFF. spam now contains all the accepted points that were
    # explored once the chain moved within min_minus_lkl - LOG_LKL_CUTOFF.
    # If the user asks for a keep_fraction <1, this is also the place where
    # a fraction (1-keep_fraction) is removed at the beginning of each chain.
    #print('--> Removing burn-in')
    spam = remove_bad_points(info)

    # Re-Map the given parameters
    info.remap_parameters(spam)

    # Now that the number of parameters is known
    # (and updated from the remap_parameters)
    # the array containing bounds can be initialised
    info.bounds = np.zeros((len(info.ref_names), len(info.levels), 2))

    # Now that the list spam contains all the different chains removed of
    # their respective burn-in, proceed to the convergence computation

    # 2D arrays for mean and var, one column will contain the total (over
    # all chains) mean (resp. variance), and each other column the
    # respective chain mean (resp. chain variance). R only contains the
    # values for each parameter. Therefore, mean and var will have len(spam)+1
    # as a first dimension
    mean = np.zeros((len(spam)+1, info.number_parameters))
    var = np.zeros((len(spam)+1, info.number_parameters))
    R = np.zeros(info.number_parameters)

    # Store the total number of points, and the total in each chain
    total = np.zeros(len(spam)+1)
    for j in xrange(len(spam)):
        total[j+1] = spam[j][:, 0].sum()
    total[0] = total[1:].sum()

    # Compute mean and variance for each chain
    print('--> Computing mean values')
    compute_mean(mean, spam, total)

    print('--> Computing variance')
    compute_variance(var, mean, spam, total)

    print('--> Computing convergence criterium (Gelman-Rubin)')
    # Gelman Rubin Diagnostic:
    # Computes a quantity linked to the ratio of the mean of the variances of
    # the different chains (within), and the variance of the means (between)
    # Note: This is not strictly speaking the Gelman Rubin test, defined for
    # same-length MC chains. Our quantity is defined without the square root,
    # which should not change much the result: a small sqrt(R) will still be a
    # small R. The same convention is used in CosmoMC, except for the weighted
    # average: we decided to do the average taking into account that longer
    # chains should count more
    within = 0
    between = 0
    for i in xrange(np.shape(mean)[1]):
        for j in xrange(len(spam)):
            within += total[j+1]*var[j+1, i]
            between += total[j+1]*(mean[j+1, i]-mean[0, i])**2
        within /= total[0]
        between /= (total[0]-1)

        R[i] = between/within
        if i == 0:
            print(' -> R-1 is %.6f' % R[i], '\tfor ', info.ref_names[i])
        else:
            print('           %.6f' % R[i], '\tfor ', info.ref_names[i])

    # Log finally the total number of steps, and absolute loglikelihood
    with open(info.log_path, 'a') as log:
        log.write("--> Total    number    of    steps: %d\n" % (
            info.steps))
        log.write("--> Total number of accepted steps: %d\n" % (
            info.accepted_steps))
        log.write("--> Minimum of -logLike           : %.2f" % (
            info.min_minus_lkl))

    # Store the remaining members in the info instance, for further writing to
    # files, storing only the mean and total of all the chains taken together
    info.mean = mean[0]
    info.R = R
    info.total = total[0]

    # Create the main chain, which consists in all elements of spam
    # put together. This will serve for the plotting.
    info.chain = np.vstack(spam)

def compute_posterior(information_instances):
    """
    computes the marginalized posterior distributions, and optionnally plots
    them

    Parameters
    ----------
    information_instances : list
        list of information objects, initialised on the given folders, or list
        of file, in input. For each of these instance, plot the 1d and 2d
        posterior distribution, depending on the flags stored in the instances,
        comming from command line arguments or read from a file.
    """
    # For convenience, store as `conf` the first element of the list
    # information_instances, since it will be called often to check for
    # configuration parameters
    conf = information_instances[0]

    # Pre configuration of the output, note that changes to the font size
    # will occur later on as well, to obtain a nice scaling.
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=11)
    matplotlib.rc('xtick', labelsize='8')
    matplotlib.rc('ytick', labelsize='8')

    # Recover max and min values for each instance, defining the a priori place
    # of ticks (in case of a comparison, this should change)
    for info in information_instances:
        info.define_ticks()
        # If plots/ folder in output folder does not exist, create it
        if os.path.isdir(os.path.join(info.folder, 'plots')) is False:
            os.mkdir(os.path.join(info.folder, 'plots'))

    # Determine the total number of parameters to plot, based on the list
    # without duplicates of the plotted parameters of all information instances
    plotted_parameters = []
    # For printing not in latex
    ref_names = []
    for info in information_instances:
        for index, name in enumerate(info.plotted_parameters):
            if name not in plotted_parameters:
                plotted_parameters.append(name)
                ref_names.append(info.ref_names[index])

    if len(plotted_parameters) == 0:
        raise io_mp.AnalyzeError(
            "You provided no parameters to analyze, probably by selecting"
            " wrong parameters names in the '--extra' file.")
    # Find the appropriate number of columns and lines for the 1d posterior
    # plot
    if  conf.num_columns_1d == None:
        num_columns = int(round(math.sqrt(len(plotted_parameters))))
    else:
        num_columns = conf.num_columns_1d
    num_lines = int(math.ceil(len(plotted_parameters)*1.0/num_columns))

    # For special needs, you can impose here a different number of columns and lines in the 1d plot
    # Here is a commented example:
    # if (len(plotted_parameters) == 10):
    #     num_columns = 5
    #     num_lines = 2

    # Create the figures
    # which will be 3*3 inches per subplot, quickly growing!
    if conf.plot:
        fig1d = plt.figure(num=1, figsize=(
            3*num_columns,
            3*num_lines), dpi=80)
    if conf.plot_2d:
        if conf.plot_diag:
            fig2d = plt.figure(num=2, figsize=(
                3*len(plotted_parameters),
                3*len(plotted_parameters)), dpi=80)
        else:
            fig2d = plt.figure(num=2, figsize=(
                3*(len(plotted_parameters)-1),
                3*(len(plotted_parameters)-1)), dpi=80)

    # Create the name of the files, concatenating the basenames with
    # underscores.
    file_name = "_".join(
        [info.basename for info in information_instances])


    # JL: READ HERE INVERSE FISHER
    if info.plot_fisher:
        try:
            # read inv_fisher file
            file_name = os.path.join(info.folder, 'inv_fisher.mat')
            n=0
            with open(file_name, 'r') as f:
                inv_fisher = np.zeros((len(info.ref_names), len(info.ref_names)), 'float64')
                for line in f:
                    if line.find('#') != -1:
                        fisher_num_param = len(line.split())-1
                        fisher_indices = np.zeros(fisher_num_param, 'int')
                        for i in range(fisher_num_param):
                            fisher_name = line.split()[i+1].replace(',', '')
                            try:
                                fisher_indices[i] = info.ref_names.index(fisher_name)
                                print('Read fisher matrix entry for parameter ',fisher_name)
                            except:
                                print('Input fisher matrix contained unknown parameter ',fisher_name)
                                fisher_indices[i] = -1
                    else:
                        if fisher_indices[n] >= 0:
                            for m in range(fisher_num_param):
                                if fisher_indices[m] >= 0:
                                    inv_fisher[fisher_indices[n],fisher_indices[m]]=line.split()[m]
                        n += 1
            #print('Read Fisher matrix:')
            #print('param  center  scale  (Fii)^1/2  (Fii)^-1/2')
            #for i in range(len(info.ref_names)):
            #    if fisher[i,i] != 0.:
            #        print(info.ref_names[i],info.centers[i],info.scales[i,i],math.sqrt(fisher[i,i]),1./math.sqrt(fisher[i,i]))
            #    else:
            #        print(info.ref_names[i],info.centers[i],' ---')
        except Warning:
            warnings.warn("Did not find inv_fisher file %s" % file_name)

    # Loop over all the plotted parameters
    # There will be two indices at all time, the one running over the plotted
    # parameters, `index`, and the one corresponding to the actual column in
    # the actual file, `native_index`. For instance, if you try to plot only
    # two columns of a several columns file, index will vary from 0 to 1, but
    # the corresponding native indices might be anything.
    # Obviously, since plotted parameters contain potentially names not
    # contained in some files (in case of a comparison), native index might be
    # undefined.
    # Defined the legends object, which will store the plot style, to display
    # at the level of the figure
    legends = [None for _ in range(len(information_instances))]
    if not conf.legendnames:
        legend_names = [info.basename.replace('_', ' ')
                        for info in information_instances]
    else:
        legend_names = conf.legendnames


    if conf.plot_2d and not conf.plot_diag:
        for info in information_instances:
            legends[info.id]=plt.Rectangle((0,0),1,1,fc =info.MP_color_cycle[info.id][1],alpha = info.alphas[info.id],linewidth=0)

    print('-----------------------------------------------')

    for index, name in enumerate(plotted_parameters):

        # Adding the subplots to the respective figures, this will correspond
        # to the diagonal on the triangle plot.
        if conf.plot_2d and conf.plot_diag:
            ax2d = fig2d.add_subplot(
                len(plotted_parameters),
                len(plotted_parameters),
                index*(len(plotted_parameters)+1)+1,
                yticks=[])
        if conf.plot:
            ax1d = fig1d.add_subplot(
                num_lines, num_columns, index+1, yticks=[])

        # check for each instance if the name is part of the list of plotted
        # parameters, and if yes, store the native_index. If not, store a flag
        # to ignore any further plotting or computing issues concerning this
        # particular instance.
        for info in information_instances:
            try:
                info.native_index = info.ref_names.index(name)
                info.ignore_param = False
                standard_name = info.backup_names[info.native_index]
            except ValueError:
                info.ignore_param = True

        # The limits might have been enforced by the user
        if name in dictkeys(conf.force_limits):
            x_span = conf.force_limits[name][1]-conf.force_limits[name][0]
            tick_min = conf.force_limits[name][0] +0.1*x_span
            tick_max = conf.force_limits[name][1] -0.1*x_span
            # [NS] We want to make sure that if there are boundaries, we don't overshoot them by the force_limits
            # (In force_limits you can only provide the limits up to the 0.1*x_span rescaling)
            bounds = info.boundaries[info.native_index]
            # Left boundary
            if bounds[0] is not None:
                if conf.force_limits[name][0] <= bounds[0]:
                    tick_min = bounds[0]
            # Right boundary
            if bounds[-1] is not None:
                if conf.force_limits[name][1] > bounds[-1]:
                    tick_max = bounds[-1]
            ticks = np.linspace(tick_min,
                                tick_max,
                                info.ticknumber)
            for info in information_instances:
                if not info.ignore_param:
                    info.x_range[info.native_index] = conf.force_limits[name]
                    info.ticks[info.native_index] = ticks
        # otherwise, find them automatically
        else:
            adjust_ticks(name, information_instances)

        print(' -> Computing histograms for ', name)
        for info in information_instances:
            if not info.ignore_param:

                # 1D posterior normalised to P_max=1 (first step)
                #
                # simply the histogram from the chains, with few bins
                #
                # normed flag deprecated since numpy v1.6, eventually not supported
                # if using a very old version replace density with normed flag
                info.hist, info.bin_edges = np.histogram(
                    # TB: without posterior_smoothing it can be nice to increase the
                    # number of bins here.
                    info.chain[:, info.native_index+2], bins=info.bins,
                    #info.chain[:, info.native_index+2], bins=2*info.bins,
                    weights=info.chain[:, 0], density=False)
                info.hist = info.hist/info.hist.max()
                # Correct for temperature
                info.hist = info.hist**conf.temperature

                info.bincenters = 0.5*(info.bin_edges[1:]+info.bin_edges[:-1])

                # 1D posterior normalised to P_max=1 (second step)
                #
                # returns a histogram still normalised to one, but with a ten times finer sampling;
                # >> first, tries a method with spline interpolation between bin centers and extrapolation at the edges
                # >> if it fails, a simpler and more robust method of linear interpolation between bin centers is used
                # >> if the interpolation module is not installed, this step keeps the same posterior
                #
                info.interp_hist, info.interp_grid = cubic_interpolation(
                    info, info.hist, info.bincenters)

                # minimum credible interval (method by Jan Hamann). Fails for
                # multimodal histograms
                bounds = minimum_credible_intervals(info)
                info.bounds[info.native_index] = bounds

        # plotting
        for info in information_instances:
            if not info.ignore_param:

                # 1D posterior normalised to P_max=1 (third step, used only for plotting)
                #
                # apply gaussian smoothing (obsolete - we don't do it anymore since the option --posterior-smoothing
                # was defined, so we commented out this part)
                #
                # factor by which the grid has been made thinner (10 means 10 times more bins)
                interpolation_factor = float(len(info.interp_grid))/float(len(info.bincenters))
                # factor for gaussian smoothing
                sigma = interpolation_factor*info.gaussian_smoothing
                # TB: A nice option when turning off posterior_smoothing is to turn on the following
                # two lines of code
                # smooth
                #smoothed_interp_hist = scipy.ndimage.filters.gaussian_filter(info.interp_hist,sigma)
                # re-normalised
                #smoothed_interp_hist = smoothed_interp_hist/smoothed_interp_hist.max()

                if conf.plot_2d and conf.plot_diag:

                    ##################################################
                    # plot 1D posterior in diagonal of triangle plot #
                    ##################################################
                    plot = ax2d.plot(
                        info.interp_grid,
                        # TB: if no posterior_smoothing uncomment/recomment the next two lines of code
                        # version without gaussian smoothing:
                        info.interp_hist,
                        # version with gaussian smoothing (commented)
                        #smoothed_interp_hist,
                        linewidth=info.line_width, ls='-',
                        color = info.MP_color_cycle[info.id][1],
                        # the [1] picks up the color of the 68% contours
                        # with [0] you would get that of the 95% contours
                        alpha = info.alphas[info.id])

                    legends[info.id] = plot[0]
                    ax2d.set_xticks(info.ticks[info.native_index])
                    if conf.legend_style == 'top':
                        ax2d.set_title(
                            '%s=$%.{0}g^{{+%.{0}g}}_{{%.{0}g}}$'.format(
                                info.decimal) % (
                                info.tex_names[info.native_index],
                                info.mean[info.native_index],
                                info.bounds[info.native_index, 0, -1],
                                info.bounds[info.native_index, 0, 0]),
                            fontsize=info.fontsize)
                        ax2d.set_xticklabels(
                            ['%.{0}g'.format(info.decimal) % s
                             for s in info.ticks[info.native_index]],
                            fontsize=info.ticksize)
                    elif conf.legend_style == 'sides':
                        # Except for the last 1d plot (bottom line), don't
                        # print(ticks)
                        if index == len(plotted_parameters)-1:
                            ax2d.set_xticklabels(
                                ['%.{0}g'.format(info.decimal) % s
                                 for s in info.ticks[info.native_index]],
                                fontsize=info.ticksize)
                            ax2d.tick_params('x',direction='inout')
                            ax2d.set_xlabel(
                                info.tex_names[info.native_index],
                                fontsize=info.fontsize)
                        else:
                            ax2d.set_xticklabels([])
                    ax2d.axis([info.x_range[info.native_index][0],
                               info.x_range[info.native_index][1],
                               0, 1.05])

                if conf.plot:
                    if conf.short_title_1d:
                        ax1d.set_title(
                            '%s'.format(info.decimal) % (
                                    info.tex_names[info.native_index]),
                            fontsize=info.fontsize)
                    else:
                        # Note the use of double curly brackets {{ }} to produce
                        # the desired LaTeX output. This is necessary because the
                        # format function would otherwise understand single
                        # brackets as fields.
                        ax1d.set_title(
                            '%s=$%.{0}g^{{+%.{0}g}}_{{%.{0}g}}$'.format(
                                info.decimal) % (
                                    info.tex_names[info.native_index],
                                    info.mean[info.native_index],
                                    info.bounds[info.native_index, 0, -1],
                                    info.bounds[info.native_index, 0, 0]),
                            fontsize=info.fontsize)

                    ax1d.set_xticks(info.ticks[info.native_index])
                    ax1d.set_xticklabels(
                        ['%.{0}g'.format(info.decimal) % s
                         for s in info.ticks[info.native_index]],
                        fontsize=info.ticksize)
                    ax1d.axis([info.x_range[info.native_index][0],
                               info.x_range[info.native_index][1],
                               0, 1.05])

                    # Execute some customisation scripts for the 1d plots
                    if (info.custom1d != []):
                        for elem in info.custom1d:
                            exec(open('plot_files/'+elem).read())

                    ##################################################
                    # plot 1D posterior in 1D plot                   #
                    ##################################################
                    ax1d.plot(
                        info.interp_grid,
                        # TB: if no posterior_smoothing uncomment/recomment the next two lines of code
                        # 1d posterior without gaussian filter:
                        info.interp_hist,
                        # gaussian filtered 1d posterior (commented):
                        #smoothed_interp_hist,
                        # raw 1d posterior:
                        #info.interp_hist,
                        lw=info.line_width, ls='-',
                        color = info.MP_color_cycle[info.id][1],
                        # the [1] picks up the color of the 68% contours
                        # with [0] you would get that of the 95% contours
                        alpha = info.alphas[info.id])
                    # uncomment if you want to see the raw points from the histogram
                    # (to check whether the inteprolation and smoothing generated artefacts)
                    #ax1d.plot(
                    #    info.bincenters,
                    #    info.hist,
                    #    'ro')

        if conf.mean_likelihood:
            for info in information_instances:
                if not info.ignore_param:
                    try:

                        # 1D mean likelihood normalised to P_max=1 (first step)
                        #
                        # simply the histogram from the chains, weighted by mutiplicity*likelihood
                        #
                        # normed flag deprecated since numpy v1.6, eventually not supported
                        # if using a very old version replace density with normed flag
                        lkl_mean, _ = np.histogram(
                            info.chain[:, info.native_index+2],
                            bins=info.bin_edges,
                            density=False,
                            weights=np.exp(
                                conf.min_minus_lkl-info.chain[:, 1])*info.chain[:, 0])
                        lkl_mean /= lkl_mean.max()

                        # 1D mean likelihood normalised to P_max=1 (second step)
                        #
                        # returns a histogram still normalised to one, but with a ten times finer sampling;
                        # >> first, tries a method with spline interpolation between bin centers and extrapolation at the edges
                        # >> if it fails, a simpler and more robust method of linear interpolation between bin centers is used
                        # >> if the interpolation module is not installed, this step keeps the same posterior
                        #
                        interp_lkl_mean, interp_grid = cubic_interpolation(
                            info, lkl_mean, info.bincenters)

                        # 1D mean likelihood normalised to P_max=1 (third step, used only for plotting)
                        #
                        # apply gaussian smoothing (obsolete - we don't do it anymore since the option --posterior-smoothing
                        # was defined, so we commented out this part)
                        #
                        # TB: if no posterior_smoothing uncomment the next two lines of code
                        # smooth
                        #smoothed_interp_lkl_mean = scipy.ndimage.filters.gaussian_filter(interp_lkl_mean,sigma)
                        # re-normalised
                        #smoothed_interp_lkl_mean = smoothed_interp_lkl_mean/smoothed_interp_lkl_mean.max()

                        # Execute some customisation scripts for the 1d plots
                        if (info.custom1d != []):
                            for elem in info.custom1d:
                                exec(open('plot_files/'+elem).read())

                        ########################################################
                        # plot 1D mean likelihood in diagonal of triangle plot #
                        ########################################################
                        if conf.plot_2d and conf.plot_diag:
                            # raw mean likelihoods:
                            #ax2d.plot(info.bincenter, lkl_mean,
                            #          ls='--', lw=conf.line_width,
                            #          color = info.MP_color_cycle[info.id][1],
                            #          alpha = info.alphas[info.id])
                            # smoothed and interpolated mean likelihoods:
                            ax2d.plot(interp_grid,
                                      # TB: if no posterior_smoothing uncomment/recomment the next two lines of code
                                      # version without gaussian smoothing:
                                      interp_lkl_mean,
                                      # version with gaussian smoothing (commented)
                                      #smoothed_interp_lkl_mean,
                                      ls='--', lw=conf.line_width,
                                      color = info.MP_color_cycle[info.id][1],
                                      alpha = info.alphas[info.id])

                        ########################################################
                        # plot 1D mean likelihood in 1D plot                   #
                        ########################################################
                        if conf.plot:
                            # raw mean likelihoods:
                            #ax1d.plot(info.bincenters, lkl_mean,
                            #          ls='--', lw=conf.line_width,
                            #          color = info.MP_color_cycle[info.id][1],
                            #          alpha = info.alphas[info.id])
                            # smoothed and interpolated mean likelihoods:
                            ax1d.plot(interp_grid,
                                      # TB: if no posterior_smoothing uncomment/recomment the next two lines of code
                                      # version without gaussian smoothing
                                      interp_lkl_mean,
                                      # version with gaussian smoothing (commented)
                                      #smoothed_interp_lkl_mean,
                                      ls='--', lw=conf.line_width,
                                      color = info.MP_color_cycle[info.id][1],
                                      alpha = info.alphas[info.id])

                    except:
                        sys.stdout.write('could not find likelihood contour for ')
                        print(info.ref_parameters[info.native_index])

        if conf.subplot is True:
            if conf.plot_2d and conf.plot_diag:
                extent2d = ax2d.get_window_extent().transformed(
                    fig2d.dpi_scale_trans.inverted())
                fig2d.savefig(os.path.join(
                    conf.folder, 'plots', file_name+'.'+conf.extension),
                    bbox_inches=extent2d.expanded(1.1, 1.4))
            if conf.plot:
                extent1d = ax1d.get_window_extent().transformed(
                    fig1d.dpi_scale_trans.inverted())
                fig1d.savefig(os.path.join(
                    conf.folder, 'plots', file_name+'.'+conf.extension),
                    bbox_inches=extent1d.expanded(1.1, 1.4))
            # Store the function in a file
            for info in information_instances:
                if not info.ignore_param:
                    hist_file_name = os.path.join(
                        info.folder, 'plots',
                        info.basename+'_%s.hist' % (
                            standard_name))
                    write_histogram(hist_file_name,
                                    info.interp_grid, info.interp_hist)

        # Now do the rest of the triangle plot
        if conf.plot_2d:
            for second_index in xrange(index):
                second_name = plotted_parameters[second_index]
                for info in information_instances:
                    if not info.ignore_param:
                        try:
                            info.native_second_index = info.ref_names.index(
                                plotted_parameters[second_index])
                            info.has_second_param = True
                            second_standard_name = info.backup_names[
                                info.native_second_index]
                        except ValueError:
                            info.has_second_param = False
                    else:
                        info.has_second_param = False

                if conf.plot_diag:
                    ax2dsub = fig2d.add_subplot(
                        len(plotted_parameters),
                        len(plotted_parameters),
                        (index)*len(plotted_parameters)+second_index+1)
                else:
                    ax2dsub = fig2d.add_subplot(
                        len(plotted_parameters)-1,
                        len(plotted_parameters)-1,
                        (index-1)*(len(plotted_parameters)-1)+second_index+1)

                for info in information_instances:
                    if info.has_second_param:

                        ax2dsub.axis([info.x_range[info.native_second_index][0],
                                      info.x_range[info.native_second_index][1],
                                      info.x_range[info.native_index][0],
                                      info.x_range[info.native_index][1]])

                        # 2D likelihood (first step)
                        #
                        # simply the histogram from the chains, with few bins only
                        #
                        # normed flag deprecated since numpy v1.6, eventually not supported
                        # if using a very old version replace density with normed flag
                        info.n, info.xedges, info.yedges = np.histogram2d(
                            info.chain[:, info.native_index+2],
                            info.chain[:, info.native_second_index+2],
                            weights=info.chain[:, 0],
                            bins=(info.bins, info.bins),
                            density=False)
                        # Correct for temperature:
                        info.n = info.n**conf.temperature

                        info.extent = [
                            info.x_range[info.native_second_index][0],
                            info.x_range[info.native_second_index][1],
                            info.x_range[info.native_index][0],
                            info.x_range[info.native_index][1]]
                        info.x_centers = 0.5*(info.xedges[1:]+info.xedges[:-1])
                        info.y_centers = 0.5*(info.yedges[1:]+info.yedges[:-1])

                        # 2D likelihood (second step)
                        #
                        # like for 1D, interpolate to get a finer grid
                        # TODO: we should not only interpolate between bin centers, but also extrapolate between side bin centers and bin edges
                        #
                        interp_y_centers = scipy.ndimage.zoom(info.y_centers,info.interpolation_smoothing, mode='reflect')
                        interp_x_centers = scipy.ndimage.zoom(info.x_centers,info.interpolation_smoothing, mode='reflect')
                        interp_likelihood = scipy.ndimage.zoom(info.n,info.interpolation_smoothing, mode='reflect')

                        # 2D likelihood (third step)
                        #
                        # gaussian smoothing
                        #
                        sigma = info.interpolation_smoothing*info.gaussian_smoothing
                        interp_smoothed_likelihood = scipy.ndimage.filters.gaussian_filter(interp_likelihood,[sigma,sigma], mode='reflect')

                        # Execute some customisation scripts for the 2d contour plots
                        if (info.custom2d != []):
                           for elem in info.custom2d:
                               exec(open('plot_files/'+elem).read())

                        # plotting contours, using the ctr_level method (from Karim
                        # Benabed). Note that only the 1 and 2 sigma contours are
                        # displayed (due to the line with info.levels[:2])
                        try:

                            ###########################
                            # plot 2D filled contours #
                            ###########################
                            if not info.contours_only:
                                contours = ax2dsub.contourf(
                                    interp_y_centers,
                                    interp_x_centers,
                                    interp_smoothed_likelihood,
                                    extent=info.extent,
                                    levels=ctr_level(
                                        interp_smoothed_likelihood,
                                        info.levels[:2]),
                                    zorder=4,
                                    colors = info.MP_color_cycle[info.id],
                                    alpha=info.alphas[info.id])

                                # now add a thin darker line
                                # around the 95% contour
                                ax2dsub.contour(
                                    interp_y_centers,
                                    interp_x_centers,
                                    interp_smoothed_likelihood,
                                    extent=info.extent,
                                    levels=ctr_level(
                                        interp_smoothed_likelihood,
                                        info.levels[1:2]),
                                    zorder=4,
                                    colors = info.MP_color_cycle[info.id][1],
                                    alpha = info.alphas[info.id],
                                    linewidths=1)

                            ###########################
                            # plot 2D contours        #
                            ###########################
                            if info.contours_only:
                                contours = ax2dsub.contour(
                                    interp_y_centers,
                                    interp_x_centers,
                                    interp_smoothed_likelihood,
                                    extent=info.extent, levels=ctr_level(
                                        interp_smoothed_likelihood,
                                        info.levels[:2]),
                                    zorder=4,
                                    colors = info.MP_color_cycle[info.id],
                                    alpha = info.alphas[info.id],
                                    linewidths=info.line_width)

                        except Warning:
                            warnings.warn(
                                "The routine could not find the contour of the " +
                                "'%s-%s' 2d-plot" % (
                                    info.plotted_parameters[info.native_index],
                                    info.plotted_parameters[info.native_second_index]))

                        # ADDING FISHER CONTOURS
                        if info.plot_fisher:
                            sub_inv_fisher = np.zeros((2,2), 'float64')
                            sub_inv_fisher[0,0] = inv_fisher[info.native_index,info.native_index]/info.scales[info.native_index,info.native_index]/info.scales[info.native_index,info.native_index]
                            sub_inv_fisher[1,1] = inv_fisher[info.native_second_index,info.native_second_index]/info.scales[info.native_second_index,info.native_second_index]/info.scales[info.native_second_index,info.native_second_index]
                            sub_inv_fisher[0,1] = inv_fisher[info.native_index,info.native_second_index]/info.scales[info.native_index,info.native_index]/info.scales[info.native_second_index,info.native_second_index]
                            sub_inv_fisher[1,0] = sub_inv_fisher[0,1]
                            if sub_inv_fisher[0,0]*sub_inv_fisher[1,1] != 0.:
                                inv_sub_inv_fisher = np.linalg.inv(sub_inv_fisher)

                                x = scipy.ndimage.zoom(info.x_centers,info.interpolation_smoothing, mode='reflect')
                                y = scipy.ndimage.zoom(info.y_centers,info.interpolation_smoothing, mode='reflect')

                                z = np.zeros((len(x),len(y)), 'float64')

                                #print(info.ref_names[info.native_index])
                                #print(info.scales)
                                #print(info.boundaries[info.native_index])
                                #print(info.centers[info.native_index])

                                for ix in range(len(x)):
                                    dx = (x[ix] - info.centers[info.native_index])
                                    for iy in range(len(y)):
                                        dy = (y[iy] - info.centers[info.native_second_index])
                                        z[ix,iy] = dx*inv_sub_inv_fisher[0,0]*dx + dy*inv_sub_inv_fisher[1,1]*dy + 2.*dx*inv_sub_inv_fisher[0,1]*dy

                                ax2dsub.contour(y,x,z,
                                    extent=info.extent,
                                    levels=[2.3,6.18],
                                    #levels=[9.30,15.79],
                                    zorder=4, colors='k')

                        ax2dsub.set_xticks(info.ticks[info.native_second_index])
                        ax2dsub.set_yticks(info.ticks[info.native_index])
                        ax2dsub.tick_params('both',direction='inout',top=True,bottom=True,left=True,right=True)
                        if index == len(plotted_parameters)-1:
                            ax2dsub.set_xticklabels(
                                ['%.{0}g'.format(info.decimal) % s for s in
                                 info.ticks[info.native_second_index]],
                                fontsize=info.ticksize)
                            if conf.legend_style == 'sides':
                                ax2dsub.set_xlabel(
                                    info.tex_names[info.native_second_index],
                                    fontsize=info.fontsize)
                        else:
                            ax2dsub.set_xticks([])

                        ax2dsub.set_yticks(info.ticks[info.native_index])
                        if second_index == 0:
                            ax2dsub.set_yticklabels(
                                ['%.{0}g'.format(info.decimal) % s for s in
                                 info.ticks[info.native_index]],
                                fontsize=info.ticksize)
                        else:
                            ax2dsub.set_yticks([])

                        if conf.legend_style == 'sides':
                            if second_index == 0:
                                ax2dsub.set_ylabel(
                                    info.tex_names[info.native_index],
                                    fontsize=info.fontsize)

                if conf.subplot is True:
                    # Store the individual 2d plots.
                    if conf.plot_2d:
                        area = ax2dsub.get_window_extent().transformed(
                            fig2d.dpi_scale_trans.inverted())
                        # Pad the saved area by 10% in the x-direction and 20% in
                        # the y-direction
                        fig2d.savefig(os.path.join(
                            conf.folder, 'plots',
                            file_name+'_2d_%s-%s.%s' % (
                                standard_name, second_standard_name,
                                conf.extension)),
                            bbox_inches=area.expanded(1.4, 1.4))

                    # store the coordinates of the points for further
                    # plotting.
                    store_contour_coordinates(
                        conf, standard_name, second_standard_name, contours)

                    for info in information_instances:
                        if not info.ignore_param and info.has_second_param:
                            info.hist_file_name = os.path.join(
                                info.folder, 'plots',
                                '{0}_2d_{1}-{2}.hist'.format(
                                    info.basename,
                                    standard_name,
                                    second_standard_name))
                    write_histogram_2d(
                        info.hist_file_name, info.x_centers, info.y_centers,
                        info.extent, info.n)

    print('-----------------------------------------------')
    if conf.plot:
        print('--> Saving figures to .{0} files'.format(info.extension))
        plot_name = '-vs-'.join([os.path.split(elem.folder)[-1]
                                for elem in information_instances])

        if conf.plot_2d:
            # Legend of triangle plot
            if ((conf.plot_legend_2d == None) and (len(legends) > 1)) or (conf.plot_legend_2d == True):
                # Create a virtual subplot in the top right corner,
                # just to be able to anchor the legend nicely
                if conf.plot_diag:
                    ax2d = fig2d.add_subplot(
                        len(plotted_parameters),
                        len(plotted_parameters),
                        len(plotted_parameters))
                else:
                    ax2d = fig2d.add_subplot(
                        len(plotted_parameters)-1,
                        len(plotted_parameters)-1,
                        len(plotted_parameters)-1)
                ax2d.axis('off')
                try:
                    ax2d.legend(legends, legend_names,
                                 loc='upper right',
                                 borderaxespad=0.,
                                 fontsize=info.legendsize)
                except TypeError:
                    ax2d.legend(legends, legend_names,
                                 loc='upper right',
                                 borderaxespad=0.,
                                 prop={'fontsize': info.legendsize})
            fig2d.subplots_adjust(wspace=0, hspace=0)
            fig2d.savefig(
                os.path.join(
                    conf.folder, 'plots', '{0}_triangle.{1}'.format(
                        plot_name, info.extension)),
                bbox_inches='tight')
        # Legend of 1D plot
        if conf.plot:
            if ((conf.plot_legend_1d == None) and (len(legends) > 1)) or (conf.plot_legend_1d == True):
                # no space left: add legend to thr right
                if len(plotted_parameters)<num_columns*num_lines:
                    fig1d.legend(legends, legend_names,
                                 loc= ((num_columns-0.9)/num_columns,0.1/num_columns),
                                 fontsize=info.legendsize)
                # space left in lower right part: add legend there
                else:
                    fig1d.legend(legends, legend_names,
                                 loc= 'center right',
                                 bbox_to_anchor = (1.2,0.5),
                                 fontsize=info.legendsize)
            fig1d.tight_layout()
            fig1d.savefig(
                os.path.join(
                    conf.folder, 'plots', '{0}_1d.{1}'.format(
                        plot_name, info.extension)),
                bbox_inches='tight')


def ctr_level(histogram2d, lvl, infinite=False):
    """
    Extract the contours for the 2d plots (Karim Benabed)

    """

    hist = histogram2d.flatten()*1.
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0]+[hist[-i] for i in alvl]+[hist.max()]
    if not infinite:
        return clist[1:]
    return clist

def minimum_credible_intervals(info):
    """
    Extract minimum credible intervals (method from Jan Haman) FIXME
    """
    histogram = info.hist
    bincenters = info.bincenters
    levels = info.levels

    bounds = np.zeros((len(levels), 2))
    j = 0
    delta = bincenters[1]-bincenters[0]
    left_edge = max(histogram[0] - 0.5*(histogram[1]-histogram[0]), 0.)
    right_edge = max(histogram[-1] + 0.5*(histogram[-1]-histogram[-2]), 0.)
    failed = False
    for level in levels:
        norm = float(
            (np.sum(histogram)-0.5*(histogram[0]+histogram[-1]))*delta)
        norm += 0.25*(left_edge+histogram[0])*delta
        norm += 0.25*(right_edge+histogram[-1])*delta
        water_level_up = np.max(histogram)*1.0
        water_level_down = np.min(histogram)*1.0
        top = 0.

        iterations = 0
        while (abs((top/norm)-level) > 0.0001) and not failed:
            top = 0.
            water_level = (water_level_up + water_level_down)/2.
            #ontop = [elem for elem in histogram if elem > water_level]
            indices = [i for i in range(len(histogram))
                       if histogram[i] > water_level]
            # check for multimodal posteriors
            if ((indices[-1]-indices[0]+1) != len(indices)):
                warnings.warn(
                    "could not derive minimum credible intervals " +
                    "for this multimodal posterior")
                warnings.warn(
                    "please try running longer chains or reducing " +
                    "the number of bins with --bins BINS (default: 20)")
                failed = True
                break
            top = (np.sum(histogram[indices]) -
                   0.5*(histogram[indices[0]]+histogram[indices[-1]]))*(delta)

            # left
            if indices[0] > 0:
                top += (0.5*(water_level+histogram[indices[0]]) *
                        delta*(histogram[indices[0]]-water_level) /
                        (histogram[indices[0]]-histogram[indices[0]-1]))
            else:
                if (left_edge > water_level):
                    top += 0.25*(left_edge+histogram[indices[0]])*delta
                else:
                    top += (0.25*(water_level + histogram[indices[0]]) *
                            delta*(histogram[indices[0]]-water_level) /
                            (histogram[indices[0]]-left_edge))

            # right
            if indices[-1] < (len(histogram)-1):
                top += (0.5*(water_level + histogram[indices[-1]]) *
                        delta*(histogram[indices[-1]]-water_level) /
                        (histogram[indices[-1]]-histogram[indices[-1]+1]))
            else:
                if (right_edge > water_level):
                    top += 0.25*(right_edge+histogram[indices[-1]])*delta
                else:
                    top += (0.25*(water_level + histogram[indices[-1]]) *
                            delta * (histogram[indices[-1]]-water_level) /
                            (histogram[indices[-1]]-right_edge))

            if top/norm >= level:
                water_level_down = water_level
            else:
                water_level_up = water_level
            # safeguard, just in case
            iterations += 1
            if (iterations > 1000):
                warnings.warn(
                    "the loop to check for sigma deviations was " +
                    "taking too long to converge")
                failed = True
                break

        # min
        if failed:
            bounds[j][0] = np.nan
        elif indices[0] > 0:
            bounds[j][0] = bincenters[indices[0]] - delta*(histogram[indices[0]]-water_level)/(histogram[indices[0]]-histogram[indices[0]-1])
        else:
            if (left_edge > water_level):
                bounds[j][0] = bincenters[0]-0.5*delta
            else:
                bounds[j][0] = bincenters[indices[0]] - 0.5*delta*(histogram[indices[0]]-water_level)/(histogram[indices[0]]-left_edge)

        # max
        if failed:
            bounds[j][1] = np.nan
        elif indices[-1] < (len(histogram)-1):
            bounds[j][1] = bincenters[indices[-1]] + delta*(histogram[indices[-1]]-water_level)/(histogram[indices[-1]]-histogram[indices[-1]+1])
        else:
            if (right_edge > water_level):
                bounds[j][1] = bincenters[-1]+0.5*delta
            else:
                bounds[j][1] = bincenters[indices[-1]] + \
                    0.5*delta*(histogram[indices[-1]]-water_level) / \
                    (histogram[indices[-1]]-right_edge)

        j += 1

    for elem in bounds:
        for j in (0, 1):
            elem[j] -= info.mean[info.native_index]
    return bounds


def write_h(info_file, indices, name, string, quantity, modifiers=None):
    """
    Write one horizontal line of output

    """
    info_file.write('\n '+name+'\t: ')
    for i in indices:
        info_file.write(string % quantity[i]+'\t')


def cubic_interpolation(info, hist, bincenters):
    """
    Small routine to accomodate the absence of the interpolate module

    """

    # we start from a try becuase if anything goes wrong, we want to return the raw histogram rather than nothing
    try:

        # test that all elements are strictly positive, otherwise we could not take the log, and we must switch to the robust method
        for i,elem in enumerate(hist):
            if elem == 0.:
                hist[i] = 1.e-99
            elif elem <0:
                print(hist[i])
                raise exception()

        # One of our methods (using polyfit) does assume that the input histogram has a maximum value of 1.
        # If in a future version this is not guaranteedanymore, we should renormalise it here.
        # This is important for computing weights and thresholds.

        # The threshold below which the likelihood will be
        # approximated as zero is hard-codeed here (could become an
        # input parameter but that would not clearly be useful).:
        threshold = 1.e-3

        # prepare the interpolation on log(Like):
        ln_hist = np.log(hist)

        # define a finer grid on a wider range (assuming that the following method is fine both for inter- and extra-polation)
        left = bincenters[0]-2.5*(bincenters[1]-bincenters[0])
        if (info.boundaries[info.native_index][0] != None):
            if (info.boundaries[info.native_index][0] > left):
                left = info.boundaries[info.native_index][0]
        right = bincenters[-1]+2.5*(bincenters[-1]-bincenters[-2])
        if (info.boundaries[info.native_index][1] != None):
            if (info.boundaries[info.native_index][1] < right):
                right = info.boundaries[info.native_index][1]
        interp_grid = np.linspace(left, right, (len(bincenters)+4)*10+1)

        ######################################
        # polynomial fit method (default):   #
        #####################################W
        if info.posterior_smoothing >= 2:
            # the points in the histogram with a very low likelihood (i.e. hist[i]<<1 hist is normalised to a maximum of one)
            # have a lot of Poisson noise and are unreliable. However, if we do nothing, they may dominate the outcome of the fitted polynomial.
            # Hence we can:
            # 1) give them less weight (weight = sqrt(hist) seems to work well)
            # 2) cut them at some threshold value and base the fit only on higher points
            # 3) both
            # the one working best seems to be 2). We also wrote 1) below, but copmmented out.

            # method 1):
            #f = np.poly1d(np.polyfit(bincenters,ln_hist,info.posterior_smoothing,w=np.sqrt(hist)))
            #interp_hist = f(interp_grid)

            # method 2):
            # find index values such that hist is negligble everywhere excepted in hist[sub_indices[0]], hist[sub_indices[-1]]
            sub_indices = [i for i,elem in enumerate(hist) if elem > threshold]
            # The interpolation is done precisely in this range: hist[sub_indices[0]] < x < hist[sub_indices[-1]]
            g = np.poly1d(np.polyfit(bincenters[sub_indices],ln_hist[sub_indices],info.posterior_smoothing)) #,w=np.sqrt(hist[sub_indices])))
            # The extrapolation is done in a range including one more bin on each side, excepted when the boundary is hit
            if (info.boundaries[info.native_index][0] == None):
                extrapolation_range_left = [bincenters[sub_indices[0]] if sub_indices[0] == 0 else bincenters[sub_indices[0]-1]]
            else:
                extrapolation_range_left = [info.boundaries[info.native_index][0] if sub_indices[0] == 0 else bincenters[sub_indices[0]-1]]
            if (info.boundaries[info.native_index][1] == None):
                extrapolation_range_right = [bincenters[sub_indices[-1]] if sub_indices[-1] == len(hist)-1 else bincenters[sub_indices[-1]+1]]
            else:
                extrapolation_range_right = [info.boundaries[info.native_index][1] if sub_indices[-1] == len(hist)-1 else bincenters[sub_indices[-1]+1]]
            # outside of this range, log(L) is brutally set to a negligible value,e, log(1.e-10)
            interp_hist = [g(elem) if (elem > extrapolation_range_left and elem < extrapolation_range_right) else np.log(1.e-10) for elem in interp_grid]

        elif info.posterior_smoothing<0:
            raise io_mp.AnalyzeError(
                "You passed --posterior-smoothing %d, this value is not understood"%info.posterior_smoothing)

        ############################################################
        # other methods:                                           #
        # - linear inter/extra-polation if posterior_smoothing = 0 #
        # - cubic inter/extra-polation if posterior_smoothing = 0  #
        ############################################################
        else:

            # try first inter/extra-polation
            try:
                # prepare to interpolate and extrapolate:
                if info.posterior_smoothing == 0:
                    f = scipy.interpolate.interp1d(bincenters, ln_hist, kind='linear', fill_value='extrapolate')
                else:
                    f = scipy.interpolate.interp1d(bincenters, ln_hist, kind='cubic', fill_value='extrapolate')
                interp_hist = f(interp_grid)

            # failure probably caused by old scipy not having the fill_value='extrapolate' argument. Then, only interpoolate.
            except:
                # define a finer grid but not a wider one
                left = bincenters[0]
                if (info.boundaries[info.native_index][0] != None):
                    if (info.boundaries[info.native_index][0] > left):
                        left = info.boundaries[info.native_index][0]
                right = bincenters[-1]
                if (info.boundaries[info.native_index][1] != None):
                    if (info.boundaries[info.native_index][1] < right):
                        right = info.boundaries[info.native_index][1]
                interp_grid = np.linspace(left, right, len(bincenters)*10+1)
                # prepare to interpolate only:
                if info.posterior_smoothing == 0:
                    f = scipy.interpolate.interp1d(bincenters, ln_hist, kind='linear')
                else:
                    f = scipy.interpolate.interp1d(bincenters, ln_hist, kind='cubic')
                interp_hist = f(interp_grid)

        # final steps used b y all methods

        # go back from ln_Like to Like
        interp_hist = np.exp(interp_hist)

        # re-normalise the interpolated curve
        interp_hist = interp_hist / interp_hist.max()

        return interp_hist, interp_grid

    except:
        # we will end up here if anything went wrong before
        # do nothing (raw histogram)
        warnings.warn(
                    "The 1D posterior could not be processed normally, probably" +
                    "due to incomplete or obsolete numpy and/or scipy versions." +
                    "So the raw histograms will be plotted.")
        return hist, bincenters

def write_histogram(hist_file_name, x_centers, hist):
    """
    Store the posterior distribution to a file

    """
    with open(hist_file_name, 'w') as hist_file:
        hist_file.write("# 1d posterior distribution\n")
        hist_file.write("\n# x_centers\n")
        hist_file.write(", ".join(
            [str(elem) for elem in x_centers])+"\n")
        hist_file.write("\n# Histogram\n")
        hist_file.write(", ".join(
            [str(elem) for elem in hist])+"\n")
    print('wrote ', hist_file_name)


def read_histogram(histogram_path):
    """
    Recover a stored 1d posterior

    """
    with open(histogram_path, 'r') as hist_file:
        for line in hist_file:
            if line:
                if line.find("# x_centers") != -1:
                    x_centers = [float(elem) for elem in
                                 hist_file.next().split(",")]
                elif line.find("# Histogram") != -1:
                    hist = [float(elem) for elem in
                            hist_file.next().split(",")]
    x_centers = np.array(x_centers)
    hist = np.array(hist)

    return x_centers, hist


def write_histogram_2d(hist_file_name, x_centers, y_centers, extent, hist):
    """
    Store the histogram information to a file, to plot it later

    """
    with open(hist_file_name, 'w') as hist_file:
        hist_file.write("# Interpolated histogram\n")
        hist_file.write("\n# x_centers\n")
        hist_file.write(", ".join(
            [str(elem) for elem in x_centers])+"\n")

        hist_file.write("\n# y_centers\n")
        hist_file.write(", ".join(
            [str(elem) for elem in y_centers])+"\n")

        hist_file.write("\n# Extent\n")
        hist_file.write(", ".join(
            [str(elem) for elem in extent])+"\n")

        hist_file.write("\n# Histogram\n")
        for line in hist:
            hist_file.write(", ".join(
                [str(elem) for elem in line])+"\n")


def read_histogram_2d(histogram_path):
    """
    Read the histogram information that was stored in a file.

    To use it, call something like this:

    .. code::

        x_centers, y_centers, extent, hist = read_histogram_2d_from_file(path)
        fig, ax = plt.subplots()
        ax.contourf(
            y_centers, x_centers, hist, extent=extent,
            levels=ctr_level(hist, [0.68, 0.95]),
            zorder=5, cma=plt.cm.autumn_r)
        plt.show()

    """
    with open(histogram_path, 'r') as hist_file:
        length = 0
        for line in hist_file:
            if line:
                if line.find("# x_centers") != -1:
                    x_centers = [float(elem) for elem in
                                 hist_file.next().split(",")]
                    length = len(x_centers)
                elif line.find("# y_centers") != -1:
                    y_centers = [float(elem) for elem in
                                 hist_file.next().split(",")]
                elif line.find("# Extent") != -1:
                    extent = [float(elem) for elem in
                              hist_file.next().split(",")]
                elif line.find("# Histogram") != -1:
                    hist = []
                    for index in range(length):
                        hist.append([float(elem) for elem in
                                     hist_file.next().split(",")])
    x_centers = np.array(x_centers)
    y_centers = np.array(y_centers)
    extent = np.array(extent)
    hist = np.array(hist)

    return x_centers, y_centers, extent, hist


def clean_conversion(module_name, tag, folder):
    """
    Execute the methods "convert" from the different sampling algorithms

    Returns True if something was made, False otherwise
    """
    has_module = False
    subfolder_name = tag+"_subfolder"
    try:
        # Don't try to import the wrong (unrequested) module, in case it's not installed
        if not folder.lower().endswith(tag.lower()):
            raise ImportError
        module = importlib.import_module(module_name)
        subfolder = getattr(module, subfolder_name)
        has_module = True
    except ImportError:
        # The module is not installed, the conversion can not take place
        pass

    if has_module and os.path.isdir(folder):
        # Remove any potential trailing slash
        folder = os.path.join(
            *[elem for elem in folder.split(os.path.sep) if elem])
        if folder.split(os.path.sep)[-1] == subfolder:
            try:
                getattr(module, 'from_%s_output_to_chains' % tag)(folder)
            except IOError:
                raise io_mp.AnalyzeError(
                    "You asked to analyze a %s folder which " % tag +
                    "seems to come from an unfinished run, or to be empty " +
                    "or corrupt. Please make sure the run went smoothly " +
                    "enough.")
            warnings.warn(
                "The content of the %s subfolder has been " % tag +
                "translated for Monte Python. Please run an "
                "analysis of the entire folder now.")
            return True
    else:
        return False


def separate_files(files):
    """
    Separate the input files in folder

    Given all input arguments to the command line files entry, separate them in
    a list of lists, grouping them by folders. The number of identified folders
    will determine the number of information instances to create
    """
    final_list = []
    temp = [files[0]]
    folder = (os.path.dirname(files[0]) if os.path.isfile(files[0])
              else files[0])
    if len(files) > 1:
        for elem in files[1:]:
            new_folder = (os.path.dirname(elem) if os.path.isfile(elem)
                          else elem)
            if new_folder == folder:
                temp.append(elem)
            else:
                folder = new_folder
                final_list.append(temp)
                temp = [elem]
    final_list.append(temp)

    return final_list


def recover_folder_and_files(files):
    """
    Distinguish the cases when analyze is called with files or folder

    Note that this takes place chronologically after the function
    `separate_files`"""
    # The following list defines the substring that a chain should contain for
    # the code to recognise it as a proper chain.
    substrings = ['.txt', '__']
    limit = 10
    # If the first element is a folder, grab all chain files inside
    if os.path.isdir(files[0]):
        folder = os.path.normpath(files[0])
        files = [os.path.join(folder, elem) for elem in os.listdir(folder)
                 if not os.path.isdir(os.path.join(folder, elem))
                 and not os.path.getsize(os.path.join(folder, elem)) < limit
                 and all([x in elem for x in substrings])]
    # Otherwise, extract the folder from the chain file-name.
    else:
        # If the name is completely wrong, say it
        if not os.path.exists(files[0]):
            raise io_mp.AnalyzeError(
                "You provided a non-existant folder/file to analyze")
        folder = os.path.relpath(
            os.path.dirname(os.path.realpath(files[0])), os.path.curdir)
        files = [os.path.join(folder, elem) for elem in os.listdir(folder)
                 if os.path.join(folder, elem) in np.copy(files)
                 and not os.path.isdir(os.path.join(folder, elem))
                 and not os.path.getsize(os.path.join(folder, elem)) < limit
                 and all([x in elem for x in substrings])]
    basename = os.path.basename(folder)
    return folder, files, basename


def extract_array(line):
    """
    Return the array on the RHS of the line

    >>> extract_array("toto = ['one', 'two']\n")
    ['one', 'two']
    >>> extract_array('toto = ["one", 0.2]\n')
    ['one', 0.2]

    """
    # Recover RHS of the equal sign, and remove surrounding spaces
    rhs = line.split('=')[-1].strip()
    # Remove array signs
    rhs = rhs.strip(']').lstrip('[')
    # Recover each element of the list
    sequence = [e.strip().strip('"').strip("'") for e in rhs.split(',')]
    for index, elem in enumerate(sequence):
        try:
            sequence[index] = int(elem)
        except ValueError:
            try:
                sequence[index] = float(elem)
            except ValueError:
                pass
    return sequence


def extract_dict(line):
    """
    Return the key and value of the dictionary element contained in line

    >>> extract_dict("something['toto'] = [0, 1, 2, -2, 'cosmo']")
    'toto', [0, 1, 2, -2, 'cosmo']

    """
    # recovering the array
    sequence = extract_array(line)
    # Recovering only the LHS
    lhs = line.split('=')[0].strip()
    # Recovering the name from the LHS
    name = lhs.split('[')[-1].strip(']')
    name = name.strip('"').strip("'")

    return name, sequence


def extract_parameter_names(info):
    """
    Reading the log.param, store in the Information instance the names
    """
    backup_names = []
    plotted_parameters = []
    boundaries = []
    ref_names = []
    tex_names = []
    scales = []
    rescales = []
    centers = []
    with open(info.param_path, 'r') as param:
        for line in param:
            if line.find('#') == -1:
                if line.find('data.experiments') != -1:
                    info.experiments = extract_array(line)
                if line.find('data.parameters') != -1:
                    name, array = extract_dict(line)
                    original = name
                    # Rename the names according the .extra file (opt)
                    if name in dictkeys(info.to_change):
                        name = info.to_change[name]
                    # If the name corresponds to a varying parameter (fourth
                    # entry in the initial array being non-zero, or a derived
                    # parameter (could be designed as fixed, it does not make
                    # any difference)), then continue the process of analyzing.
                    if array[3] != 0 or array[5] == 'derived' or array[5] == 'derived_lkl':
                        # The real name is always kept, to have still the class
                        # names in the covmat
                        backup_names.append(original)
                        # With the list "to_plot", we can potentially restrict
                        # the variables plotted. If it is empty, though, simply
                        # all parameters will be plotted.
                        if info.to_plot == []:
                            plotted_parameters.append(name)
                        else:
                            if name in info.to_plot:
                                plotted_parameters.append(name)

                        # Append to the boundaries array
                        boundaries.append([
                            None if elem == 'None' or (isinstance(elem, int)
                                                       and elem == -1)
                            else elem for elem in array[1:3]])
                        ref_names.append(name)
                        # Take care of the scales
                        scale = array[4]
                        rescale = 1.
                        if name in dictkeys(info.new_scales):
                            rescale = info.new_scales[name]/array[4]
                        scales.append(scale)
                        rescales.append(rescale)

                        # Given the scale, decide for the pretty tex name
                        number = 1./(scale*rescale)
                        tex_names.append(
                            io_mp.get_tex_name(name, number=number))

                        # Read starting values (useful for plotting Fisher)
                        centers.append(array[0])

    scales = np.diag(scales)
    rescales = np.diag(rescales)

    info.ref_names = ref_names
    info.tex_names = tex_names
    info.boundaries = boundaries
    info.backup_names = backup_names
    info.scales = scales
    info.rescales = rescales
    # Beware, the following two numbers are different. The first is the total
    # number of parameters stored in the chain, whereas the second is for
    # plotting purpose only.
    info.number_parameters = len(ref_names)
    info.plotted_parameters = plotted_parameters

    info.centers = centers


def find_maximum_of_likelihood(info):
    """
    Finding the global maximum of likelihood

    min_minus_lkl will be appended with all the maximum likelihoods of files,
    then will be replaced by its own maximum. This way, the global
    maximum likelihood will be used as a reference, and not each chain's
    maximum.
    """
    min_minus_lkl = []
    for chain_file in info.files:
        # cheese will brutally contain everything (- log likelihood) in the
        # file chain_file being scanned.
        # This could potentially be faster with pandas, but is already quite
        # fast
        #
        # This would read the chains including comment lines:
        #cheese = (np.array([float(line.split()[1].strip())
        #                    for line in open(chain_file, 'r')]))
        #
        # This reads the chains excluding comment lines:
        with open(chain_file, 'r') as f:
            cheese = (np.array([float(line.split()[1].strip())
                                for line in py_filterfalse(iscomment,f)]))

        try:
            min_minus_lkl.append(cheese[:].min())
        except ValueError:
            pass
    # beware, it is the min because we are talking about
    # '- log likelihood'
    # Selecting only the true maximum.
    try:
        min_minus_lkl = min(min_minus_lkl)
    except ValueError:
        raise io_mp.AnalyzeError(
            "No decently sized chain was found in the desired folder. " +
            "Please wait to have more accepted point before trying " +
            "to analyze it.")

    info.min_minus_lkl = min_minus_lkl


def remove_bad_points(info):
    """
    Create an array with all the points from the chains, after removing non-markovian, burn-in and fixed fraction

    """
    # spam will brutally contain all the chains with sufficient number of
    # points, after the burn-in was removed.
    spam = list()

    # Recover the longest file name, for pleasing display
    max_name_length = max([len(e) for e in info.files])

    # Total number of steps done:
    steps = 0
    accepted_steps = 0

    # Open the log file
    log = open(info.log_path, 'w')

    for index, chain_file in enumerate(info.files):
        # To improve presentation, and print only once the full path of the
        # analyzed folder, we recover the length of the path name, and
        # create an empty complementary string of this length
        total_length = 18+max_name_length
        empty_length = 18+len(os.path.dirname(chain_file))+1

        basename = os.path.basename(chain_file)
        if index == 0:
            exec("sys.stdout.write('--> Scanning file %-{0}s' % chain_file)".format(
                max_name_length))
        else:
            exec("sys.stdout.write('%{0}s%-{1}s' % ('', basename))".format(
                empty_length, total_length-empty_length))
        # cheese will brutally contain everything in the chain chain_file being
        # scanned
        #
        # This would read the chains including comment lines:
        #cheese = (np.array([[float(elem) for elem in line.split()]
        #                    for line in open(chain_file, 'r')]))
        #
        # This read the chains excluding comment lines:
        with open(chain_file, 'r') as f:
            cheese = (np.array([[float(elem) for elem in line.split()]
                                for line in py_filterfalse(iscomment,f)]))
        # If the file contains a broken line with a different number of
        # elements, the previous array generation might fail, and will not have
        # the correct shape. Hence the following command will fail. To avoid
        # that, the error is caught.
        try:
            local_min_minus_lkl = cheese[:, 1].min()
        except IndexError:
            raise io_mp.AnalyzeError(
                "Error while scanning %s." % chain_file +
                " This file most probably contains "
                "an incomplete line, rendering the analysis impossible. "
                "I think that the following line(s) is(are) wrong:\n %s" % (
                    '\n '.join(
                        ['-> %s' % line for line in
                         open(chain_file, 'r') if
                         len(line.split()) != len(info.backup_names)+2])))
        line_count = float(sum(1 for line in open(chain_file, 'r')))

        # Logging the information obtained until now.
        number_of_steps = cheese[:, 0].sum()
        log.write("%s\t " % os.path.basename(chain_file))
        log.write(" Number of steps:%d\t" % number_of_steps)
        log.write(" Steps accepted:%d\t" % line_count)
        log.write(" acc = %.2g\t" % (float(line_count)/number_of_steps))
        log.write("min(-loglike) = %.2f\n" % local_min_minus_lkl)
        steps += number_of_steps
        accepted_steps += line_count

        # check if analyze() is called directly by the user, or by the mcmc loop during an updating phase
        try:
            # command_line.update is defined when called by the mcmc loop
            info.update
        except:
            # in case it was not defined (i.e. when analyze() is called directly by user), set it to False
            info.update = 0
        # check if analyze() is called directly by the user, or by the mcmc loop during an updating phase
        try:
            # command_line.adaptive is defined when called by the mcmc loop
            info.adaptive
        except:
            # in case it was not defined (i.e. when analyze() is called directly by user), set it to False
            info.adaptive = 0

        # Removing non-markovian part, burn-in, and fraction= (1 - keep-fraction)
        start = 0
        markovian=0
        try:
            # Read all comments in chains about times when proposal was updated
            # The last of these comments gives the number of lines to be skipped in the files
            if info.markovian and not info.update:
                with open(chain_file, 'r') as f:
                    for line in py_filter(iscomment,f):
                        if info.only_markovian or ('update proposal' in line):
                            start = int(line.split()[2])
                        else:
                            pass
                markovian = start

            # Remove burn-in, defined as all points until the likelhood reaches min_minus_lkl+LOG_LKL_CUTOFF
            # except when it is run in adaptive mode
            if not info.adaptive:
                while cheese[start, 1] > info.min_minus_lkl+LOG_LKL_CUTOFF:
                        start += 1
            burnin = start-markovian

            # Remove fixed fraction as requested by user (usually not useful if non-markovian is also removed)
            if info.keep_fraction < 1:
                start = start + int((1.-info.keep_fraction)*(line_count - start))

            sys.stdout.write(": Removed ")
            if info.markovian:
                sys.stdout.write("%d non-markovian points, " % markovian)
            sys.stdout.write("%d points of burn-in, " % burnin)
            if info.keep_fraction < 1:
                sys.stdout.write("and first %.0f percent, " % (100.*(1-info.keep_fraction)))
            print("keep %d steps" % (line_count-start))

        except IndexError:
            print(': Removed everything: chain not converged')


        # ham contains cheese without the burn-in, if there are any points
        # left (more than 5)
        if np.shape(cheese)[0] > start+5:
            ham = np.copy(cheese[int(start)::])

            # Deal with single file case
            if len(info.files) == 1:
                warnings.warn("Convergence computed for a single file")
                bacon = np.copy(ham[::3, :])
                egg = np.copy(ham[1::3, :])
                sausage = np.copy(ham[2::3, :])

                spam.append(bacon)
                spam.append(egg)
                spam.append(sausage)
                continue

            # Adding resulting table to spam
            spam.append(ham)

    # Test the length of the list
    if len(spam) == 0:
        raise io_mp.AnalyzeError(
            "No decently sized chain was found. " +
            "Please wait a bit to analyze this folder")

    # Applying now new rules for scales, if the name is contained in the
    # referenced names
    for name in dictkeys(info.new_scales):
        try:
            index = info.ref_names.index(name)
            for i in xrange(len(spam)):
                spam[i][:, index+2] *= 1./info.rescales[index, index]
        except ValueError:
            # there is nothing to do if the name is not contained in ref_names
            pass

    info.steps = steps
    info.accepted_steps = accepted_steps

    return spam


def compute_mean(mean, spam, total):
    """
    """
    for i in xrange(np.shape(mean)[1]):
        for j in xrange(len(spam)):
            submean = np.sum(spam[j][:, 0]*spam[j][:, i+2])
            mean[j+1, i] = submean / total[j+1]
            mean[0, i] += submean
        mean[0, i] /= total[0]


def compute_variance(var, mean, spam, total):
    """
    """
    for i in xrange(np.shape(var)[1]):
        for j in xrange(len(spam)):
            var[0, i] += np.sum(
                spam[j][:, 0]*(spam[j][:, i+2]-mean[0, i])**2)
            var[j+1, i] = np.sum(
                spam[j][:, 0]*(spam[j][:, i+2]-mean[j+1, i])**2) / \
                (total[j+1]-1)
        var[0, i] /= (total[0]-1)


def compute_covariance_matrix(info):
    """
    """
    covar = np.zeros((len(info.ref_names), len(info.ref_names)))
    for i in xrange(len(info.ref_names)):
        for j in xrange(i, len(info.ref_names)):
            covar[i, j] = (
                info.chain[:, 0]*(
                    (info.chain[:, i+2]-info.mean[i]) *
                    (info.chain[:, j+2]-info.mean[j]))).sum()
            if i != j:
                covar[j, i] = covar[i, j]
    covar /= info.total

    # Removing scale factors in order to store true parameter covariance
    covar = np.dot(info.scales.T, np.dot(covar, info.scales))

    return covar


def adjust_ticks(param, information_instances):
    """
    """
    if len(information_instances) == 1:
        return
    # Recovering all x_range and ticks entries from the concerned information
    # instances
    x_ranges = []
    ticks = []
    for info in information_instances:
        if not info.ignore_param:
            x_ranges.append(info.x_range[info.native_index])
            ticks.append(info.ticks[info.native_index])

    # The new x_range and tick should min/max all the existing ones
    new_x_range = np.array(
        [min([e[0] for e in x_ranges]), max([e[1] for e in x_ranges])])

    temp_ticks = np.array(
        [min([e[0] for e in ticks]), max([e[-1] for e in ticks])])

    new_ticks = np.linspace(temp_ticks[0],
                            temp_ticks[1],
                            info.ticknumber)

    for info in information_instances:
        if not info.ignore_param:
            info.x_range[info.native_index] = new_x_range
            info.ticks[info.native_index] = new_ticks


def store_contour_coordinates(info, name1, name2, contours):
    """docstring"""
    file_name = os.path.join(
        info.folder, 'plots', '{0}_2d_{1}-{2}.dat'.format(
            info.basename, name1, name2))

    with open(file_name, 'w') as plot_file:
        plot_file.write(
            '# contour for confidence level {0}\n'.format(
                info.levels[1]))
        for elem in contours.collections[0].get_paths():
            points = elem.vertices
            for k in range(np.shape(points)[0]):
                plot_file.write("%.8g\t %.8g\n" % (
                    points[k, 0], points[k, 1]))
                # stop to not include the inner contours
                if k != 0:
                    if all(points[k] == points[0]):
                        plot_file.write("\n")
                        break
        plot_file.write("\n\n")
        plot_file.write(
            '# contour for confidence level {0}\n'.format(
                info.levels[0]))
        for elem in contours.collections[1].get_paths():
            points = elem.vertices
            for k in range(np.shape(points)[0]):
                plot_file.write("%.8g\t %.8g\n" % (
                    points[k, 0], points[k, 1]))
                if k != 0:
                    if all(points[k] == points[0]):
                        plot_file.write("\n")
                        break
        plot_file.write("\n\n")

def iscomment(s):
    """
    Define what we call a comment in MontePython chain files
    """
    return s.startswith('#')

class Information(object):
    """
    Hold all information for analyzing runs

    """
    # Counting the number of instances, to choose the color map
    _ids = count(0)
    # Flag checking the absence or presence of the interp1d function
    has_interpolate_module = False

    # Actual pairs of colors used by MP.
    # For each pair, the first color is for the 95% contour,
    # and the second for the 68% contour + the 1d probability.
    # Note that, as with the other customisation options, you can specify new
    # values for this in the extra plot_file.
    MP_color = {
        'Red':['#E37C80','#CE121F'],
        'Blue':['#7A98F6','#1157EF'],
        'Green':['#88B27A','#297C09'],
        'Orange':['#F3BE82','#ED920F'],
        'Grey':['#ABABAB','#737373'],
        'Purple':['#B87294','#88004C']
    }
    # order used when several directories are analysed
    MP_color_cycle = [
        MP_color['Red'],
        MP_color['Blue'],
        MP_color['Green'],
        MP_color['Orange'],
        MP_color['Grey'],
        MP_color['Purple']
    ]
    # in the same order, list of transparency levels
    alphas = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    def __init__(self, command_line, other=None):
        """
        The following initialization creates the three tables that can be
        customized in an extra plot_file (see :mod:`parser_mp`).

        Parameters
        ----------
        command_line : Namespace
            it contains the initialised command line arguments
        """
        self.to_change = {}
        """
        Dictionary whose keys are the old parameter names, and values are the
        new ones. For instance :code:`{'beta_plus_lambda':'beta+lambda'}`

        """
        self.to_derive = {}
        """
        Dictionary whose keys are new parameter names and values are formulas to calculate them.
        For instance :code:`{'beta_plus_lambda':'beta+lambda'}`

        """
        self.to_reorder = []
        """
        Array of names to re-order in the plotting. Names not included in this list
        will appear at the end in their usual ordering

        """
        self.to_plot = []
        """
        Array of names of parameters to plot. If left empty, all will be
        plotted.

        .. warning::
            If you changed a parameter name with :attr:`to_change`, you need to
            give the new name to this array

        """
        self.new_scales = {}
        """
        Dictionary that redefines some scales. The keys will be the parameter
        name, and the value its scale.

        """

        # Assign a unique id to this instance
        self.id = next(self._ids)

        # Defining the sigma contours (1, 2 and 3-sigma, and 95% CL)
        self.levels = np.concatenate([scipy.special.erf(np.array([1, 2, 3])/np.sqrt(2)),[0.95]])

        # Follows a bunch of initialisation to provide default members
        self.ref_names, self.backup_names = [], []
        self.scales, self.plotted_parameters = [], []
        self.spam = []

        # Store directly all information from the command_line object into this
        # instance, except the protected members (begin and end with __)
        for elem in dir(command_line):
            if elem.find('__') == -1:
                setattr(self, elem, getattr(command_line, elem))

        # initialise the legend flags
        self.plot_legend_1d = None
        self.plot_legend_2d = None

        # initialize the legend size to be the same as fontsize, but can be
        # altered in the extra file
        self.legendsize = self.fontsize
        self.legendnames = []

        # initialize the customisation script flags
        self.custom1d = []
        self.custom2d = []

        # initialise the dictionary enmforcing limit
        self.force_limits = {}

        # Read a potential file describing changes to be done for the parameter
        # names, and number of paramaters plotted (can be let empty, all will
        # then be plotted), but also the style of the plot. Note that this
        # overrides the command line options
        if command_line.optional_plot_file:
            plot_file_vars = {'info': self,'plt': plt}
            exec(open(command_line.optional_plot_file).read(), plot_file_vars)

        # check and store keep_fraction
        if command_line.keep_fraction<=0 or command_line.keep_fraction>1:
            raise io_mp.AnalyzeError("after --keep-fraction you should pass a float >0 and <=1")
        self.keep_fraction = command_line.keep_fraction

    def remap_parameters(self, spam):
        """
        Perform substitutions of parameters for analyzing

        .. note::

            for arbitrary combinations of parameters, the prior will not
            necessarily be flat.

        """
        if hasattr(self, 'redefine'):
            for key, value in dictitems(self.redefine):
                # Check that the key was an original name
                if key in self.backup_names:
                    print(' /|\  Transforming', key, 'into', value)
                    # We recover the indices of the key
                    index_to_change = self.backup_names.index(key)+2
                    print('/_o_\ The new variable will be called ' +
                          self.ref_names[self.backup_names.index(key)])
                    # Recover all indices of all variables present in the
                    # remapping
                    variable_names = [elem for elem in self.backup_names if
                                      value.find(elem) != -1]
                    indices = [self.backup_names.index(name)+2 for name in
                               variable_names]
                    # Now loop over all files in spam
                    for i in xrange(len(spam)):
                        # Assign variables to their values
                        for index, name in zip(indices, variable_names):
                            exec("%s = spam[i][:, %i]" % (name, index))
                        # Assign to the desired index the combination
                        exec("spam[i][:, %i] = %s" % (index_to_change, value))
        if hasattr(self, 'to_derive'):
            for key, value in dictitems(self.to_derive):
                print(' /|\  Creating new parameter', key)
                print('/_o_\ with formula ' + key + " = "+value)
                # Recover all indices of all variables present in the
                # remapping
                variable_names = [elem for elem in self.backup_names if
                                  value.find(elem) != -1]
                indices = [self.backup_names.index(name)+2 for name in
                           variable_names]
                # Now loop over all files in spam
                for i in xrange(len(spam)):
                    # For each file expand the dimension of spam by one
                    spam[i] = np.hstack([spam[i],np.empty((len(spam[i]),1))])
                    # Assign local variables to their values
                    for index, name in zip(indices, variable_names):
                        exec("%s = spam[i][:, %i]" % (name, index))
                    # Assign to to the appended array the combination
                    exec("spam[i][:,-1] = %s" % (value))

                # If everything was successfull, add the corresponding info
                self.ref_names.append(key)
                self.tex_names.append(io_mp.get_tex_name(key,number=1))
                self.backup_names.append(key)
                self.boundaries.append([None,None])
                N = len(self.scales)
                self.scales = np.vstack([np.hstack([self.scales,np.zeros((N,1))]),np.zeros((N+1,1)).T])
                self.scales[-1,-1]=1
                self.rescales = np.vstack([np.hstack([self.rescales,np.zeros((N,1))]),np.zeros((N+1,1)).T])
                self.rescales[-1,-1]=1
                self.number_parameters +=1
                self.plotted_parameters.append(key)
                self.centers = np.append(self.centers,0)
        if hasattr(self, 'to_reorder'):
            if(len(self.to_reorder)>0):
                indices = [self.backup_names.index(name) for name in self.to_reorder]
                missing_indices = [x for x in np.arange(len(self.backup_names)) if x not in indices]
                indices = np.concatenate([indices,missing_indices],dtype=int)
                self.ref_names = [self.ref_names[i] for i in indices]
                self.tex_names = [self.tex_names[i] for i in indices]
                self.backup_names = [self.backup_names[i] for i in indices]
                self.boundaries = [self.boundaries[i] for i in indices]
                self.scales = self.scales[indices][:,indices]
                self.rescales = self.rescales[indices][:,indices]
                self.centers = self.centers[indices]
                # Re-sort spam (barely any overhead, due to numpy's internal memory views)
                for i in xrange(len(spam)):
                    spam[i][:,2:] = spam[i][:,indices+2]
                # Play the same game independently for plotted_parameters
                # since these might be a lot fewer
                indices = [self.plotted_parameters.index(name) for name in self.to_reorder]
                if(len(indices)>0):
                  missing_indices = [x for x in np.arange(len(self.plotted_parameters)) if x not in indices]
                  indices = np.concatenate([indices,missing_indices])
                  self.plotted_parameters = [self.plotted_parameters[i] for i in indices]

    def define_ticks(self):
        """
        """
        self.max_values = self.chain[:, 2:].max(axis=0)
        self.min_values = self.chain[:, 2:].min(axis=0)
        self.span = (self.max_values-self.min_values)
        # Define the place of ticks, given the number of ticks desired, stored
        # in conf.ticknumber
        self.ticks = np.array(
            [np.linspace(self.min_values[i]+self.span[i]*0.1,
                         self.max_values[i]-self.span[i]*0.1,
                         self.ticknumber) for i in range(len(self.span))])
        # Define the x range (ticks start not exactly at the range boundary to
        # avoid display issues)
        self.x_range = np.array((self.min_values, self.max_values)).T

        # In case the exploration hit a boundary (as defined in the parameter
        # file), at the level of precision defined by the number of bins, the
        # ticks and x_range should be altered in order to display this
        # meaningful number instead.
        for i in range(np.shape(self.ticks)[0]):
            x_range = self.x_range[i]
            bounds = self.boundaries[i]
            # Left boundary
            if bounds[0] is not None:
                if abs(x_range[0]-bounds[0]) < self.span[i]/self.bins:
                    self.ticks[i][0] = bounds[0]
                    self.x_range[i][0] = bounds[0]
            # Right boundary
            if bounds[-1] is not None:
                if abs(x_range[-1]-bounds[-1]) < self.span[i]/self.bins:
                    self.ticks[i][-1] = bounds[-1]
                    self.x_range[i][-1] = bounds[-1]

    def write_information_files(self):

        # Store in info_names only the tex_names that were plotted, for this
        # instance, and in indices the corresponding list of indices. It also
        # removes the $ signs, for clarity
        self.info_names = [
            name for index, name in enumerate(self.tex_names) if
            self.ref_names[index] in self.plotted_parameters]
        self.indices = [self.tex_names.index(name) for name in self.info_names]
        self.tex_names = [name for index, name in enumerate(self.tex_names) if
            self.ref_names[index] in self.plotted_parameters]
        self.info_names = [name.replace('$', '') for name in self.info_names]

        # Define the bestfit array
        self.bestfit = np.zeros(len(self.ref_names))
        for i in xrange(len(self.ref_names)):
            self.bestfit[i] = self.chain[self.sorted_indices[0], :][2+i]

        # Write down to the .h_info file all necessary information
        self.write_h_info()
        self.write_v_info()
        self.write_tex()

    def write_h_info(self):

        with open(self.h_info_path, 'w') as h_info:
            h_info.write(' param names\t:  ')
            for name in self.info_names:
                h_info.write("%-14s" % name)

            write_h(h_info, self.indices, 'R-1 values', '% .6f', self.R)
            write_h(h_info, self.indices, 'Best Fit  ', '% .6e', self.bestfit)
            write_h(h_info, self.indices, 'mean      ', '% .6e', self.mean)
            write_h(h_info, self.indices, 'sigma     ', '% .6e',
                    (self.bounds[:, 0, 1]-self.bounds[:, 0, 0])/2.)
            h_info.write('\n')
            write_h(h_info, self.indices, '1-sigma - ', '% .6e',
                    self.bounds[:, 0, 0])
            write_h(h_info, self.indices, '1-sigma + ', '% .6e',
                    self.bounds[:, 0, 1])
            write_h(h_info, self.indices, '2-sigma - ', '% .6e',
                    self.bounds[:, 1, 0])
            write_h(h_info, self.indices, '2-sigma + ', '% .6e',
                    self.bounds[:, 1, 1])
            write_h(h_info, self.indices, '3-sigma - ', '% .6e',
                    self.bounds[:, 2, 0])
            write_h(h_info, self.indices, '3-sigma + ', '% .6e',
                    self.bounds[:, 2, 1])

            # bounds
            h_info.write('\n')
            write_h(h_info, self.indices, '1-sigma > ', '% .6e',
                    self.mean+self.bounds[:, 0, 0])
            write_h(h_info, self.indices, '1-sigma < ', '% .6e',
                    self.mean+self.bounds[:, 0, 1])
            write_h(h_info, self.indices, '2-sigma > ', '% .6e',
                    self.mean+self.bounds[:, 1, 0])
            write_h(h_info, self.indices, '2-sigma < ', '% .6e',
                    self.mean+self.bounds[:, 1, 1])
            write_h(h_info, self.indices, '3-sigma > ', '% .6e',
                    self.mean+self.bounds[:, 2, 0])
            write_h(h_info, self.indices, '3-sigma < ', '% .6e',
                    self.mean+self.bounds[:, 2, 1])
            write_h(h_info, self.indices, '95% > ', '% .6e',
                    self.mean+self.bounds[:, -1, 0])
            write_h(h_info, self.indices, '95% < ', '% .6e',
                    self.mean+self.bounds[:, -1, 1])

    def write_v_info(self):
        """Write vertical info file"""
        with open(self.v_info_path, 'w') as v_info:
            v_info.write('%-15s\t:  %-11s' % ('param names', 'R-1'))
            v_info.write(' '.join(['%-11s' % elem for elem in [
                'Best fit', 'mean', 'sigma', '1-sigma -', '1-sigma +',
                '2-sigma -', '2-sigma +', '1-sigma >', '1-sigma <',
                '2-sigma >', '2-sigma <', '95% CL >', '95% CL <']]))
            for index, name in zip(self.indices, self.info_names):
                v_info.write('\n%-15s\t: % .4e' % (name, self.R[index]))
                v_info.write(' '.join(['% .4e' % elem for elem in [
                    self.bestfit[index], self.mean[index],
                    (self.bounds[index, 0, 1]-self.bounds[index, 0, 0])/2.,
                    self.bounds[index, 0, 0], self.bounds[index, 0, 1],
                    self.bounds[index, 1, 0], self.bounds[index, 1, 1],
                    self.mean[index]+self.bounds[index, 0, 0],
                    self.mean[index]+self.bounds[index, 0, 1],
                    self.mean[index]+self.bounds[index, 1, 0],
                    self.mean[index]+self.bounds[index, 1, 1],
                    self.mean[index]+self.bounds[index, -1, 0],
                    self.mean[index]+self.bounds[index, -1, 1]]]))

    def write_tex(self):
        """Write a tex table containing the main results """
        with open(self.tex_path, 'w') as tex:
            tex.write("\\begin{tabular}{|l|c|c|c|c|} \n \\hline \n")
            tex.write("Param & best-fit & mean$\pm\sigma$ ")
            tex.write("& 95\% lower & 95\% upper \\\\ \\hline \n")
            for index, name in zip(self.indices, self.tex_names):
                tex.write("%s &" % name)
                tex.write("$%.4g$ & $%.4g_{%.2g}^{+%.2g}$ " % (
                    self.bestfit[index], self.mean[index],
                    self.bounds[index, 0, 0], self.bounds[index, 0, 1]))
                tex.write("& $%.4g$ & $%.4g$ \\\\ \n" % (
                    self.mean[index]+self.bounds[index, -1, 0],
                    self.mean[index]+self.bounds[index, -1, 1]))

            tex.write("\\hline \n \\end{tabular} \\\\ \n")
            tex.write("$-\ln{\cal L}_\mathrm{min} =%.6g$, " % (
                self.min_minus_lkl))
            tex.write("minimum $\chi^2=%.4g$ \\\\ \n" % (
                self.min_minus_lkl*2.))
