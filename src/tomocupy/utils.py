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
import subprocess



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
            
           
def downsampleZarr(data, scale_factor=2, max_levels=1):
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
        new_shape = tuple(max(1, dim // 1) for dim in current_level.shape)
        if min(new_shape) <= 1:
            break
        current_level = current_level#resize(current_level, new_shape, order=0, preserve_range=True, anti_aliasing=True)
        print(current_level.shape)
        levels.append(current_level)
    return levels            
    
    
def clean_zarr(output_path):
    if os.path.exists(output_path):
        try:
            subprocess.run(["mv", output_path, output_path + "_tmp"], check=True)
            subprocess.run(["ls -lrt", output_path + "_tmp"], check=True)
            subprocess.run(["rm", "-rf", output_path + "_tmp"], check=True)
            log.info(f"Successfully removed directory: {output_path}")
        except subprocess.CalledProcessError as e:
            log.error(f"Error removing directory {output_path}: {e}")
            raise
    else:
        log.warning(f"Path does not exist: {output_path}")      
