.. _setup:

Getting Started
===============
brownian_ot is designed to run using Python 3 and Matlab.
See the list of required Python packages in `requirements.txt`.

The easiest way to install an up-to-date release of Python along with
the most widely-used packages for scientific computing is by installing the
`Anaconda distribution <https://www.anaconda.com/products/individual>`_.

Installing other requirements
-----------------------------
Here are other things you will need to install to access the full functionality of brownian_ot:

 #. brownian_ot uses `quaternions <https://en.wikipedia.org/wiki/Quaternion>`_ to handle arbitrary rotations in 3D. There is a package, `quaternion`, that extends NumPy to handle quaternions. It lives on GitHub `here <https://github.com/moble/quaternion>`_. See its README for installation instructions, or `its page on conda-forge <https://anaconda.org/conda-forge/quaternion>`_.
 #. You will need a working current installation of Matlab.
 #. You will also need to install the `Matlab Engine API for Python <https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`_.
 #. Download (preferably, via `git clone`) `ott <https://github.com/ilent2/ott>`_ in order to calculate optical forces.  
 #. In order to calculate optical forces on sphere clusters, you need to build `mstm <http://www.eng.auburn.edu/~dmckwski/scatcodes/>`_, which means that you will also need to install a Fortran compiler. The open-source `gfortran <https://gcc.gnu.org/fortran/>`_ compiler, part of the `GNU Compiler Collection (gcc) <https://gcc.gnu.org/>`_, works well. Unless you know what you are doing, it is easiest to install binaries rather than building from source. See the suggestions `here <https://gcc.gnu.org/wiki/GFortranBinaries>`_, as well as the links below:
 
     * Linux: use your distribution's package manager.
     * Windows: consider the binaries from MinGW (see `installation notes <http://www.mingw.org/wiki/Getting_Started>`_.)
     * Mac: if you have Homebrew or MacPorts installed, use that. Otherwise, see instructions above.
       
 #. See instructions for compiling mstm in the `manual <http://www.eng.auburn.edu/~dmckwski/scatcodes/mstm-manual-2013-v3.0.pdf>`_.


Configuration
-------------

The template configuration file `paths_config.yaml.rename` is at the top level of the package. Copy this file and save it in the same location as `paths_config.yaml` (the code looks for a file with this specific name). You will need to change the paths to the following:
 
     * The folder in which you saved ott.
     * The folder `brownian_ot/matlab/` wherever you saved brownian_ot.
     * The executable for mstm.

       
Running unit tests and building documentation
---------------------------------------------
       
To run the unit tests for this package (which will confirm that everything was installed correctly), you will need to install `pytest <https://docs.pytest.org/en/latest/>`_. This can be done via anaconda. To actually run the tests, which should take a minute or two, navigate to the root brownian_ot folder and run ::

       python -m pytest

To build the documentation, you will need to install `Sphinx <https://www.sphinx-doc.org/en/master/>`_. This can also be done via anaconda. Once you have done so, navigate to the `docs/` folder and run ::

       make html

to build html pages you can navigate in a web browser. (See the Sphinx documentation for other build options).
   

  
