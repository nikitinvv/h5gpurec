#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

from pathlib import Path
import numpy as np
import h5py
import cupy as cp
import argparse
from threading import Thread
import time
import numexpr as ne
import sys
import os
import tifffile as tiff
from skimage.transform import resize
import time
from functools import wraps




from tomocupy import logging
log = logging.getLogger(__name__)

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def printProgressBar(iteration, total, qsize, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(
        f'\rqueue size {qsize:03d} | {prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def positive_int(value):
    """Convert *value* to an integer and make sure it is positive."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('Only positive integers are allowed')
    return result


def restricted_float(x):

    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def pinned_array(array):
    """Allocate pinned memory and associate it with numpy array"""

    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(
        mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


def signal_handler(sig, frame):
    """Calls abort_scan when ^C or ^Z is typed"""

    print('Abort')
    sys.exit(1)


class WRThread():
    def __init__(self):
        self.thread = None

    def run(self, fun, args):
        self.thread = Thread(target=fun, args=args)
        self.thread.start()

    def is_alive(self):
        if self.thread == None:
            return False
        return self.thread.is_alive()

    def join(self):
        if self.thread == None:
            return
        self.thread.join()


def find_free_thread(threads):
    ithread = 0
    while True:
        if not threads[ithread].is_alive():
            break
        ithread = ithread+1
        # ithread=(ithread+1)%len(threads)
        if ithread == len(threads):
            ithread = 0
            time.sleep(0.01)
    return ithread


def downsample(data, binning):
    """Downsample data"""
    for j in range(binning):
        x = data[:, :, ::2]
        y = data[:, :, 1::2]
        data = ne.evaluate('x + y')  # should use multithreading
    for k in range(binning):
        x = data[:, ::2]
        y = data[:, 1::2]
        data = ne.evaluate('x + y')
    return data


def _copy(res, u, st, end):
    res[st:end] = u[st:end]


def copy(u, res, nthreads=16):
    nchunk = int(np.ceil(u.shape[0]/nthreads))
    mthreads = []
    for k in range(nthreads):
        th = Thread(target=_copy, args=(
            res, u, k*nchunk, min((k+1)*nchunk, u.shape[0])))
        mthreads.append(th)
        th.start()
    for th in mthreads:
        th.join()
    return res


def _copyTransposed(res, u, st, end):
    res[st:end] = u[:, st:end].swapaxes(0, 1)


def copyTransposed(u, res=[], nthreads=16):
    if res == []:
        res = np.empty([u.shape[1], u.shape[0], u.shape[2]], dtype=u.dtype)
    nchunk = int(np.ceil(u.shape[1]/nthreads))
    mthreads = []
    for k in range(nthreads):
        th = Thread(target=_copyTransposed, args=(
            res, u, k*nchunk, min((k+1)*nchunk, u.shape[1])))
        mthreads.append(th)
        th.start()
    for th in mthreads:
        th.join()
    return res


def read_bright_ratio(params):
    '''Read the ratio between the bright exposure and other exposures.
    '''
    log.info('  *** *** Find bright exposure ratio params from the HDF file')
    try:
        possible_names = ['/measurement/instrument/detector/different_flat_exposure',
                          '/process/acquisition/flat_fields/different_flat_exposure']
        for pn in possible_names:
            if check_item_exists_hdf(params.file_name, pn):
                diff_bright_exp = param_from_dxchange(params.file_name, pn,
                                                      attr=None, scalar=False, char_array=True)
                break
        if diff_bright_exp.lower() == 'same':
            log.error('  *** *** used same flat and data exposures')
            params.bright_exp_ratio = 1
            return params
        possible_names = ['/measurement/instrument/detector/exposure_time_flat',
                          '/process/acquisition/flat_fields/flat_exposure_time',
                          '/measurement/instrument/detector/brightfield_exposure_time']
        for pn in possible_names:
            if check_item_exists_hdf(params.file_name, pn):
                bright_exp = param_from_dxchange(params.file_name, pn,
                                                 attr=None, scalar=True, char_array=False)
                break
        log.info('  *** *** %f' % bright_exp)
        norm_exp = param_from_dxchange(params.file_name,
                                       '/measurement/instrument/detector/exposure_time',
                                       attr=None, scalar=True, char_array=False)
        log.info('  *** *** %f' % norm_exp)
        params.bright_exp_ratio = bright_exp / norm_exp
        log.info(
            '  *** *** found bright exposure ratio of {0:6.4f}'.format(params.bright_exp_ratio))
    except:
        log.warning('  *** *** problem getting bright exposure ratio.  Use 1.')
        params.bright_exp_ratio = 1
    return params


def check_item_exists_hdf(hdf_filename, item_name):
    '''Checks if an item exists in an HDF file.
    Inputs
    hdf_filename: str filename or pathlib.Path object for HDF file to check
    item_name: name of item whose existence needs to be checked
    '''
    with h5py.File(hdf_filename, 'r') as hdf_file:
        return item_name in hdf_file


def param_from_dxchange(hdf_file, data_path, attr=None, scalar=True, char_array=False):
    """
    Reads a parameter from the HDF file.
    Inputs
    hdf_file: string path or pathlib.Path object for the HDF file.
    data_path: path to the requested data in the HDF file.
    attr: name of the attribute if this is stored as an attribute (default: None)
    scalar: True if the value is a single valued dataset (dafault: True)
    char_array: if True, interpret as a character array.  Useful for EPICS strings (default: False)
    """
    if not Path(hdf_file).is_file():
        return None
    with h5py.File(hdf_file, 'r') as f:
        try:
            if attr:
                return f[data_path].attrs[attr].decode('ASCII')
            elif char_array:
                return ''.join([chr(i) for i in f[data_path][0]]).strip(chr(0))
            elif scalar:
                return f[data_path][0]
            else:
                return None
        except KeyError:
            return None
            
            
def progress_bar(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        total = kwargs.pop('total', 100)  # Total number of iterations (default is 100)
        bar_length = 40  # Length of the progress bar

        for i in range(total):
            result = func(*args, **kwargs)  # Call the original function
            percent_complete = (i + 1) / total
            bar = '#' * int(percent_complete * bar_length) + '-' * (bar_length - int(percent_complete * bar_length))
            sys.stdout.write(f'\rProgress: [{bar}] {percent_complete:.2%}')
            sys.stdout.flush()
            time.sleep(0.01)  # Simulate work being done
        
        print("\nCompleted!")
        return result
    
    return wrapper


@progress_bar
def calculate_global_min_max(input_dir, bin_factor=4):
    """
    Calculate the global minimum and maximum pixel values of all TIFF images in a directory after downsampling (binning).

    Parameters:
    - input_dir (str): Path to the directory containing TIFF images.
    - bin_factor (int, optional): Factor by which to downsample the images before calculating min and max values. Default is 4.

    Returns:
    - global_min (float): The minimum pixel value across all images.
    - global_max (float): The maximum pixel value across all images.

    Raises:
    - ValueError: If no TIFF files are found in the directory.
    """
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tiff','.tif'))])
    if not file_list:
        raise ValueError(f"No TIFF files found in the directory: {input_dir}")

    global_min = np.inf
    global_max = -np.inf
    
    for file in file_list:
        image = tiff.imread(file)
        binned_image = resize(image, 
                              (image.shape[0] // bin_factor, image.shape[1] // bin_factor), 
                              order=1, preserve_range=True, anti_aliasing=False).astype(image.dtype)
        global_min = min(global_min, binned_image.min())
        global_max = max(global_max, binned_image.max())
    
    #info(f"Global min and max found: {global_min}, dtype: {global_max}")
    return global_min, global_max
      
    

def minmaxHisto(input_dir, thr=1e-5, num_bins=1000):
    # Read the image files
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tiff','.tif'))])
    if not file_list:
        raise ValueError(f"No TIFF files found in the directory: {input_dir}")

    # Choose the middle image from the list
    middle_image_path = file_list[len(file_list) // 2]

    # Read the image
    image = tiff.imread(middle_image_path)
    if image is None:
        raise ValueError(f"Image not found at {middle_image_path}")

    # Calculate the histogram
    hist, bin_edges = np.histogram(image, bins=num_bins)

    # Find the start and end indices based on a threshold
    threshold = np.max(hist) * thr
    stend = np.where(hist > threshold)
    if len(stend[0]) == 0:
        raise ValueError("No significant histogram bins found.")

    st = stend[0][0]
    end = stend[0][-1]

    # Determine min and max values
    mmin = bin_edges[st]
    mmax = bin_edges[end + 1]

    # Ensure min and max are not too close
    if np.isclose(mmin, mmax):
        raise ValueError("The minimum and maximum values are too close. Adjust the threshold or bin count.")

    return mmin, mmax

    
       
def load_tiff_chunked(input_dir, dtype, chunk_size, start_index=0, global_min=None, global_max=None):
    """
    Load TIFF images from a directory in chunks and convert them to a specified data type, optionally normalizing the values.

    Parameters:
    - input_dir (str): Path to the directory containing TIFF images.
    - dtype (numpy dtype): Target data type for the output array.
    - chunk_size (int): Number of images to load in each chunk.
    - start_index (int, optional): Starting index for loading images. Default is 0.
    - global_min (float, optional): Minimum pixel value for normalization. If None, no normalization is performed. Default is None.
    - global_max (float, optional): Maximum pixel value for normalization. If None, no normalization is performed. Default is None.

    Returns:
    - zarr_chunk (numpy array): A chunk of loaded images converted to the specified data type.
    - end_index (int): The end index for the current chunk of images.

    Raises:
    - ValueError: If no TIFF files are found in the directory.
    """
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tiff','.tif'))])
    if not file_list:
        raise ValueError(f"No TIFF files found in the directory: {input_dir}")

    end_index = min(start_index + chunk_size, len(file_list))
    chunk_files = file_list[start_index:end_index]
    if not chunk_files:
        return None, end_index

    sample_image = tiff.imread(chunk_files[0])
    chunk_shape = (len(chunk_files),) + sample_image.shape

    zarr_chunk = np.zeros(chunk_shape, dtype=dtype)
    
    for i, file in enumerate(chunk_files):
        image = tiff.imread(file)
        if global_min is not None and global_max is not None:
            image = (image - global_min) / (global_max - global_min)  # Normalize to [0, 1]
            if dtype == np.uint16 or dtype == np.int16:
                image = image * (2**15 - 1)  # Scale to int16 range
            elif dtype == np.uint8 or dtype == np.int8:
                image = image * (2**8 - 1)  # Scale to int8 range

        zarr_chunk[i] = image.astype(dtype)
        
    #info(f"Loaded TIFF chunk with shape: {zarr_chunk.shape}, dtype: {zarr_chunk.dtype}")
    return zarr_chunk, end_index

def downsampleZarr(data, scale_factor=2, max_levels=6):
    """
    Create a multi-level downsampled version of the input data.

    Parameters:
    - data (numpy array): The input image data to be downsampled.
    - scale_factor (int, optional): Factor by which to downsample the data at each level. Default is 2.
    - max_levels (int, optional): Maximum number of downsampled levels to generate. Default is 6.

    Returns:
    - levels (list of numpy arrays): A list containing the original data and each downsampled level.

    Logs:
    - Information about the shape and data type of each downsampled level.
    """
    current_level = data
    levels = [current_level]
    for _ in range(max_levels):
        new_shape = tuple(max(1, dim // 2) for dim in current_level.shape)
        if min(new_shape) <= 1:
            break
        current_level = resize(current_level, new_shape, order=0, preserve_range=True, anti_aliasing=True)
        levels.append(current_level)
    return levels            
