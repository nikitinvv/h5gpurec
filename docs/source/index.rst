.. image:: img/tomocupy-logo-wide.svg
   :width: 320px
   :alt: TomoPy Logo
   :align: center

|

========
Tomocupy
========

**Tomocupy** is a Python package and a command-line interface for GPU reconstruction of tomographic/laminographic data in 16-bit and 32-bit precision. All preprocessing operations are implemented on GPU with using CuPy library, the backprojection operation is implemented with CUDA C.

Tomocupy implements an efficient data processing conveyor allowing to overlap all data transfers with computations. First, independent Python threads are started for reading data chunks from the hard disk into a Python data queue and for writing reconstructed chunks from the Python queue to the hard disk. Second, CPU-GPU data transfers are overlapped with GPU computations by using CUDA streams. 

**Note:** For the best performance it is recommended to use fast data storage otherwise most of reconstruction time will be spent for rw operations. PCIe NVME SSDs (e.g., Samsung 970/980 EVO/PRO) allows for parallel data read/write operations and with tomocupy works 4-8 times faster than regular HDD. 

Features
--------

* Fast tomographic reconstruction on GPU
* Fast laminographic reconstruction on GPU
* Efficient conveyor data processing
* 16-bit or 32-bit arithmetics
* Manual rotation center search and automatic rotation center search with using SIFT algorithm
* Manual search of the laminographic tilting angle
* Additional preprocessing steps: ring removal with wavelets, phase retrieval procedure with Paganin filter, dezinger filtering
* Back-projection is implemented with 3 different methods (Fourier-based, Log-polar-based, and direct discretization of the line intergral)
* Saving data in tiff and h5 formats



Contribute
----------

* Documentation: https://github.com/tomography/tomocupy/tree/master/doc
* Issue Tracker: https://github.com/tomography/tomocupy/docs/issues
* Source Code: https://github.com/tomography/tomocupy/

Content
-------

.. toctree::
   :maxdepth: 2

   install
   usage
   performance
   tested_gpus
   api
   credits
