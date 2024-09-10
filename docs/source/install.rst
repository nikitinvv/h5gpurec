=====================
Installation and test
=====================

Tomocupy works in NVidia GPUs of compute capability 6.0 and higher. To run tomocupy the system should have an `nvidia driver installed <https://developer.nvidia.com/cuda-downloads>`_. Cuda Toolkit is not necessary.
Run ``nvidia-smi`` to check whether the driver is installed. For check compute capability of the GPU, see `this document <http://mylifeismymessage.net/find-the-compute-capability-of-your-nvidia-graphics-card-gpu/>`_. 

0. For faster installation of packages, it is better to use `Miniforge <https://github.com/conda-forge/miniforge>`_. Then go to Step 3.

1. For regular Anaconda, add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict


2. Environmental solver mamba works much faster than the regular one, use

::

    (base)$ conda install -n base conda-libmamba-solver
    (base)$ conda config --set solver libmamba

3. Create environment with installed tomocupy

::

    (base)$ conda create -n tomocupy tomocupy

4. Activate tomocupy environment

::

    (base)$ conda activate tomocupy
    

5. Test installation

::

    (tomocupy)$ tomocupy recon -h

============================
Installation for development
============================

0. For faster installation of packages, it is better to use `Miniforge <https://github.com/conda-forge/miniforge>`_. Then go to Step 3.

1. Add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict

2. Environmental solver mamba works much faster than the regular one, use

::

    (base)$ conda install -n base conda-libmamba-solver
    (base)$ conda config --set solver libmamba

3. Create environment with necessary dependencies

::

    (base)$ conda create -n tomocupy -c conda-forge cupy scikit-build swig numexpr opencv tifffile h5py cmake pywavelets python=3.10


.. warning:: Conda has a built-in mechanism to determine and install the latest version of cudatoolkit supported by your driver. However, if for any reason you need to force-install a particular CUDA version (say 11.0), you can do:
  
  conda install -c conda-forge cupy cudatoolkit=11.0
  

4. Activate tomocupy environment

::

    (base)$ conda activate tomocupy

5*. (If needed) Install meta for supporting hdf meta data writer used by option: --save-format h5

::

    (tomocupy)$ git clone https://github.com/xray-imaging/meta.git
    (tomocupy)$ cd meta
    (tomocupy)$ pip install .
    (tomocupy)$ cd -


6. Make sure that the path to nvcc compiler is set (or set it by e.g. ``export CUDACXX=/local/cuda-11.7/bin/nvcc``) and install tomocupy

::
    
    (tomocupy)$ git clone https://github.com/tomography/tomocupy
    (tomocupy)$ cd tomocupy
    (tomocupy)$ pip install .

===================================
Additional instructions for Windows
===================================

#. Install `Build VS 2019 utils <https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2019>`_

#. Install CUDA toolkit, e.g. `cuda 11_2_2 <https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64>`_

#. Install `Anaconda for windows <https://docs.anaconda.com/free/anaconda/install/windows/>`_ and use Powershell in which tomocupy environment can be created

.. note::
    It is better to have only one version of VS and one version of CUDA toolkit on your system to avoid problems with environmental variables

==========
Unit tests
==========
Check the library path to cuda or set it by ``export LD_LIBRARY_PATH=/local/cuda-11.7/lib64``

Run the following to check all functionality
::

    (tomocupy)$ cd tests; bash test_all.sh


Update
======

**tomocupy** is constantly updated to include new features. To update your locally installed version

::

    (tomocupy)$ cd tomocupy
    (tomocupy)$ git pull
    (tomocupy)$ pip install .



Installation on Polaris supercomputer
=====================================
1. connect to Polaris main node (computing nodes don't have access to the internet)  and install anaconda

2. add modules:

::

    module add gcc/11.2.0
    module add cudatoolkit-standalone/11.4.4

.. note::
    We work with cuda-11.4 not with cuda-12.1 because the current driver version on polaris is 11.4

3. create tomocupy environment, specifying cudatoolkit=11.4

::

    conda create -n tomocupy -c conda-forge cupy scikit-build swig numexpr opencv tifffile h5py cmake cudatoolkit=11.4

4. clone tomocupy:

::

    git clone https://github.com/tomography/tomocupy

5. install tomocupy

::

    cd tomocupy; pip install .

6. test tomocupy:

:: 

    tomocupy recon -h

7. connect to a node with GPUs in interactive mode and a debug allocation for now, smth like

::

    qsub -I -A hp-ptycho -l select=4:system=polaris -l filesystems=home:eagle -l walltime=30:00 -q debug-scaling

.. note::
    Replace hp-ptycho by your project

8. test tomocupy:

::

    cd tests; bash test_all.sh
