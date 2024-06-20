===========================================================
Monte Python, a Monte Carlo Markov Chain code (with Class!)
===========================================================

:Main developer: Thejs Brinckmann <thejs.brinckmann@gmail.com>
:Author: Benjamin Audren <benjamin.audren@epfl.ch>
:License: MIT


The code is under the MIT license. As an additional clause, when using the code
in a scientific publication you are also required to cite the v3.0 release paper
``MontePython 3: boosted MCMC sampler and other features`` and the original release
paper ``Conservative Constraints on Early Cosmology`` (see the tail of this document
for the bibtex entries).

Recent changelog
----------------
v3.6.1, June 20, 2024

* Added euclid_spectroscopic and euclid_photometric_alm likelihoods from 2303.09451, 2405.06047 (S. Casas, M. Doerenkamp, J. Lesgourgues, S. Pamuk, L. Rathmann, Sabarish V., N. Schoeneberg)

v3.6.0, June 1, 2023

* Added Lyman-alpha alpha-beta-gamma-delta likelihood from 2206.08188 (D. C. Hooper, N. Schoeneberg, R. Murgia)

* Added eBOSS DR16 BAO-only likelihoods (S. Alvi, T. Brinckmann)

* Added eBOSS DR16 BAO-plus likelihoods (V. Poulin, T. Brinckmann)

* Added Pantheon+ and Pantheon+SH0ES likelihoods (V. Poulin)

* Added KiDS-1000 likelihoods (B. Stoelzner)

* Added S_8 as sampling parameter (B. Stoelzner, T. Brinckmann)

* Various bugfixes and pull requests

v3.5.0, Sep 20, 2021

* Added framework for spectral distortions likelihoods, plus mock FIRAS and PIXIE (D. C. Hooper, M. Lucca, N. Schoeneberg)

* Added new Pantheon likelihood with prior on the absolute magnitude (D. Camarena)

* Updated Pantheon likelihood to latest value

* Various bugfixes related to python2 / python3 compatibility (D. C. Hooper, N. Schoeneberg)

* Various additional bugfixes

v3.4.0, Feb 25, 2021

* Added BBN likelihoods from 1907.11594 (N. Schoeneberg, D. C. Hooper)

* Added H0LICOW likelihoods from 1907.04869 (S. Taubenberger, S. Suyu)

* New feature: experiment specifications passed in .param files now overwrite ones in .data files, which allows for testing different likelihood specifications simultaneously (N. Schoeneberg)

* New plotting features: to_derive allows to add new parameters in the plot file without overwriting the old parameter, and to_reorder allows to choose the order of parameters for the plots (N. Schoeneberg)

* Significant speed-up of Pantheon likelihood (J. Renk)

* Improved treatment of tick limits, sigma bounds, and Omega_m / omega_m (N. Schoeneberg)

* Various bugfixes related to python2 / python3 compatibility (D. C. Hooper, N. Schoeneberg, F. Agocs, J. Renk)

* Various additional bugfixes


Details and Examples
--------------------

If you are searching for further details or specific examples of a work session,
please refer to the online documentation. See also the `Monte Python 3 paper
<https://arxiv.org/abs/1804.07261>`_ for details on the code, including a
summary of features as of v3.0.

Note the `Monte Python 3 paper <https://arxiv.org/abs/1804.07261>`_ contains an
overview of all likelihoods currently implemented in the code, with some details
on those likelihoods, such as datasets, last updated, type and relevant papers
to cite when using the likelihood. In the future, the overview of likelihoods
will be maintained on the official `Monte Python website
<https://brinckmann.github.io/montepython_public/>`_.

You can find installation details below and on the archived `Monte Python 2 wiki
<https://github.com/baudren/montepython_public/wiki>`_. The `Monte Python 3 forum
<https://github.com/brinckmann/montepython_public/issues>`_ contains a
collection of already answered questions, and can be used to discuss the code.
Also refer to the archived `Monte Python 2 forum
<https://github.com/baudren/montepython_public/issues>`_ for additional
previously answered questions, but please post all new issues on the
`Monte Python 3 forum <https://github.com/brinckmann/montepython_public/issues>`_.

The official `Monte Python website
<https://brinckmann.github.io/montepython_public/>`_, the
`course page of Julien Lesgourgues <https://lesgourg.github.io/courses.html>`_,
and the `hi_class website <http://miguelzuma.github.io/hi_class_public>`_ contain *Monte Python*
(and *Class*) lectures, examples and exercises.


Want to contribute?
-------------------

*Monte Python* is developed and maintained by volunteer workers and we are always
happy for new people to contribute. Do not hesitate to contact us if you believe
you have something to add, this can be e.g. new likelihoods, new samplers,
improvements to the plotting, bug fixes, or ideas for how to improve the code.
Additionally, everyone is encouraged to assist in resolving issues on the forum,
so do not hesitate to reply if you think you can help.

In particular, if you would like to have your likelihood added to the public
github repository, please make sure it is well documented and add all relevant
information to the .data file, e.g. authors and references.


Prerequisites
-------------

* You need the python program **version 2.7.x** or **version 3.x**.
  Note that lower versions of python may work, down to 2.6, if you
  add manually two extra packages
  (`ordereddict <http://code.activestate.com/recipes/576693/>`_ and
  `argparse <https://pypi.python.org/pypi/argparse/1.2.1>`_).

* Your python of choice must have `numpy` (version >= 1.4.1) and `cython`. The
  later is used to wrap CLASS in python.

* *[optional]* If you want to use fully the plotting capabilities of Monte Python,
  you also need the `scipy`, with interpolate, and `matplotlib` modules.

* *[optional]* You can now use Multi Nest and the CosmoHammer with Monte
  Python, though you need to install them. Please refer to the documentation.


The MontePython part
--------------------

Move the `.tar.bz2` file to the place of your convenience, untar its content

.. code::

    $ bunzip2 montepython-vx.y.tar.bz2
    $ tar -xvf montepython-vx.y.tar

This will create a directory named `montepython` into your current directory.
You can add the following line to your `.bashrc` file:

.. code::

    export PATH=/path/to/MontePython/montepython/:$PATH

to be able to call the program from anywhere.

You will need to adapt only two files to your local configuration. The first
is the main file of the code `montepython/MontePython.py`, and it will be the only
time you will have to edit it, and it is simply to accommodate different
possible configurations of your computer.

Its first line reads

.. code::

    #!/usr/bin/python

This should be changed to wherever is your preferred python distribution
installed. For standard distribution, this should already be working. Now,
you should be able to execute directly the file, i.e. instead of calling:

The second file to modify is located in the root directory of Monte Python :
`default.conf`. This file will be read (and stored) whenever you execute the
program, and will search for your cosmological code path, your data path, and
your wmap wrapper path. You can alternatively create a second one, `my.conf`,
containing your setup, and then run the code providing this file (with the flag
`--conf`)


The Class part
--------------

Go to your class directory, and do **make clean**, then **make**. This builds the
`libclass.a`, needed for the next step. From there,

.. code::

    $ cd python/
    $ python setup.py build
    $ python setup.py install --user

This will compile the file `classy.pyx`, which is the python wrapper for CLASS,
into a library, `classy.so`, located in the `build/` subdirectory. This is the
library called in Monte Python afterwards.

If this step fails, check that you have `cython` installed, `numpy` (a numerical
package for python), python (well... did I say this code was in python ?) with
a version > 2.6.  If this step fails again, kindly ask your system admin, (s)he
is there for this, after all. Note that the installation (last command) is
not strictly speaking mandatory.

Take care to use the same Python version when compiling CLASS as will be used
when running Monte Python.

Remember that if you modify `CLASS` to implement some new physics, you will need to
perform this part again for the new `CLASS`.


The Planck likelihood part
---------------------------

*Written by Deanna C. Hooper* <hooper@physik.rwth-aachen.de>

The Planck 2018 data can be found on the `Planck Legacy Archive <http://pla.esac.esa.int/pla/#home>`_.
The Planck Likelihood Code (**plc**) is based on a library called `clik`. It will be extracted,
alongside several `.clik` folders that contain the likelihoods. The code uses an auto installer device,
called `waf`. Here we detail the full installation.

Move to the directory where you want Planck 2018

.. code::

   $ cd path/to/planck

Download the code and baseline data (will need 300 Mb of space)

.. code::

    $ wget -O COM_Likelihood_Code-v3.0_R3.01.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Code-v3.0_R3.01.tar.gz"
    $ wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz"

Uncompress the code and the likelihood, and do some clean-up

.. code::

    $ tar -xvzf COM_Likelihood_Code-v3.0_R3.01.tar.gz
    $ tar -xvzf COM_Likelihood_Data-baseline_R3.00.tar.gz
    $ rm COM_Likelihood_*tar.gz

Move into the code directory

.. code::

    $ cd code/plc_3.0/plc-3.01

Configure the code. Note that you are **strongly advised** to configure clik with the Intel mkl library, and not with lapack.
There is a massive gain in execution time: without it, the code is dominated by the execution of the low-l polarisation data.
Before the next step make sure you do NOT have any old Planck likelihoods sourced!

.. code::

   $ ./waf configure --lapack_mkl=${MKLROOT} --install_all_deps

If everything went well, you are ready to install the code

.. code::

   $ ./waf install

You now need to source the likelihood. If you are running on a bash shell, simply type

.. code::

   $ source bin/clik_profile.sh

If you are running in a z-shell, you will first need to create a .zsh version of the above file. This can be done in many ways, for example

.. code::

   $ cp bin/clik_profile.sh bin/clik_profile.zsh
   $ sed -i 's/addvar PATH /PATH=$PATH:/g' bin/clik_profile.zsh
   $ sed -i 's/addvar PYTHONPATH /PYTHONPATH=$PYTHONPATH:/g' bin/clik_profile.zsh
   $ sed -i 's/addvar LD_LIBRARY_PATH /LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/g' bin/clik_profile.zsh
   $ source bin/clik_profile.zsh

You need to add 'source /path/to/planck/code/plc_3.0/plc-3.01/bin/clik_profile.sh' to your .bashrc (or the .zsh to your
.zshrc on a z-shell), and you should put it in your scripts for cluster computing.

In your *Monte Python* configuration file, you will need to add

.. code::

   path['clik'] = '/path/to/planck/code/plc_3.0/plc-3.01'

There are nine Planck 2018 likelihoods defined in *Monte Python*: `Planck_highl_TT`, `Planck_highl_TT_lite`,
`Planck_highl_TTTEEE`, `Planck_highl_TTTEEE_lite`, `Planck_lensing`, `Planck_lowl_TT`, `Planck_lowl_EE`,
`Planck_lowl_EEBB`, `Planck_lowl_BB`, as well as five sets of parameter files, bestfit files, and covmats.


Enjoying the difference
-----------------------

Now the code is installed. Go anywhere, and just call

.. code::

    $ python montepython/MontePython.py --help
    $ python montepython/MontePython.py run --help
    $ python montepython/MontePython.py info --help

To see a list of all commands. For the `run` subcommand, there are two
essential ones, without which the program will not start. At minimum, you
should precise an output folder (`-o`) and a parameter file (`-p`). An example
of parameter file is found in the main directory of MontePython (`test.param`,
for instance).

A typical call would then be:

.. code::

    $ python montepython/MontePython.py run -o test -p example.param

If non existent, the `test/` folder will be created, and a run with the number
of steps described in `example.param` will be started. To run a chain with more
steps, one can type:

.. code::

    $ python montepython/MontePython.py run -o test -p example.param -N 100

If you want to analyse the run, then just type

.. code::

    $ python montepython/MontePython.py info test/

Note that you probably want more than a hundred points before analyzing a
folder.


Bibtex entry
------------

When using *Monte Python* in a publication, please acknowledge the code by citing
the following papers. If you used *Class*, *MultiNest*, *PolyChord* or *Cosmo Hammer*,
you should also cite the original works.

Please also cite the relevant papers for each likelihood used: as of v3.0 we have a
list of references for all likelihoods in the first of the papers below. In the
future the list will be maintained on the official `Monte Python website
<https://brinckmann.github.io/montepython_public/>`_. Otherwise, this information can
often be found in the .data file of the likelihood folder.

In order to encourage people to both develop and share likelihoods with the community,
to the benefit of all users, we optionally encourage users to cite the paper in which
the *Monte Python* likelihood was first used, in addition to the papers in which data
and/or likelihoods were published.

.. code::

    @article{Brinckmann:2018cvx,
          author         = "Brinckmann, Thejs and Lesgourgues, Julien",
          title          = "{MontePython 3: boosted MCMC sampler and other features}",
          year           = "2018",
          eprint         = "1804.07261",
          archivePrefix  = "arXiv",
          primaryClass   = "astro-ph.CO",
          SLACcitation   = "%%CITATION = ARXIV:1804.07261;%%"
    }
    @article{Audren:2012wb,
          author         = "Audren, Benjamin and Lesgourgues, Julien and Benabed,
                            Karim and Prunet, Simon",
          title          = "{Conservative Constraints on Early Cosmology: an
                            illustration of the Monte Python cosmological parameter
                            inference code}",
          journal        = "JCAP",
          volume         = "1302",
          pages          = "001",
          doi            = "10.1088/1475-7516/2013/02/001",
          year           = "2013",
          eprint         = "1210.7183",
          archivePrefix  = "arXiv",
          primaryClass   = "astro-ph.CO",
          reportNumber   = "CERN-PH-TH-2012-290, LAPTH-048-12",
          SLACcitation   = "%%CITATION = ARXIV:1210.7183;%%",
    }
