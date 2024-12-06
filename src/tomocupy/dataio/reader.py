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

from tomocupy import logging
from tomocupy import utils
from tomocupy.global_vars import args, params
from ast import literal_eval

import numpy as np
import h5py

from threading import Thread

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Reader', ]

log = logging.getLogger(__name__)


class Reader():
    '''
    Class for reading APS DXfiles and configure array sizes as required by tomocupy
    '''

    def __init__(self):

        if args.dark_file_name == None:
            args.dark_file_name = args.file_name
        else:
            log.warning(f'Using dark fields from {args.dark_file_name}')

        if args.flat_file_name == None:
            args.flat_file_name = args.file_name
        else:
            log.warning(f'Using flat fields from {args.flat_file_name}')

        self.init_sizes()
        if args.reconstruction_type[:3] == 'try':
            self.init_sizes_try()
        if args.lamino_angle != 0:
            self.init_sizes_lamino()

    def init_sizes(self):
        """Calculating and adjusting sizes for reconstruction by chunks"""

        # read data sizes and projection angles with a reader
        sizes = self.read_sizes()
        theta = self.read_theta(sizes['nproji'])
        nproji = sizes['nproji']
        nzi = sizes['nzi']
        ni = sizes['ni']
        nflat = sizes['nflat']
        ndark = sizes['ndark']
        dtype = sizes['dtype']

        # adjust data type for input data
        if args.binning > 0:
            in_dtype = args.dtype
        else:
            in_dtype = dtype

        # adjust the last row and proj ids, so as first and last corrdinates in x
        if (args.end_row == -1):
            args.end_row = nzi
        if (args.end_proj == -1):
            args.end_proj = nproji
        if (args.end_column == -1):
            args.end_column = ni

        # find numebr of rows
        nz = args.end_row-args.start_row

        # define z chunk size for processing
        ncz = args.nsino_per_chunk
        if ncz == 1 and args.reconstruction_algorithm == 'fourierrec':
            ncz = 2  # 2 rows are processed at the same time since fourierrec works with complex numbers

        # define projection chunk size for processing
        ncproj = args.nproj_per_chunk

        centeri = args.rotation_axis
        if centeri == -1:
            centeri = ni/2

        st_n = args.start_column
        end_n = args.end_column

        # work only with multiples of 2
        nsh = (end_n-st_n) % (2**(args.binning+1))
        if nsh != 0:
            end_n -= nsh
            log.warning(
                f'Decreasing projection width by {nsh} pixel to operate with multiple of 2 sizes. New projection width: {end_n-st_n}')

        ni = end_n - st_n
        centeri -= st_n

        # update sizes wrt binning
        ni //= 2**args.binning
        centeri /= 2**args.binning
        nz //= 2**args.binning

        # change sizes for 360 deg scans with rotation axis at the border
        if (args.file_type == 'double_fov'):
            n = 2*ni
            if (centeri < ni//2):
                # if rotation center is on the left side of the ROI
                center = ni-centeri
            else:
                center = centeri
        else:
            n = ni
            center = centeri

        # adjust sizes for 16bit processing
        if args.dtype == 'float16':
            center += (2**int(np.log2(ni))-ni)/2
            st_n = (ni-2**int(np.log2(ni)))//2
            end_n = st_n+2**int(np.log2(ni))
            ni = 2**int(np.log2(ni))
            n0 = n
            n = 2**int(np.log2(n))

            if n != n0:
                log.warning(
                    f'Crop data to the power of 2 sizes to work with 16bit precision, output size in x dimension {ni}')

        # blocked views fix
        ids_proj = [args.start_proj, args.end_proj]
        theta = theta[ids_proj[0]:ids_proj[1]]
        if args.blocked_views != 'none':
            tmp = literal_eval(args.blocked_views)
            if not isinstance(tmp[0], list):
                tmp = [tmp]
            ids = np.arange(len(theta))
            for pairs in tmp:
                [st, end] = pairs
                ids = np.intersect1d(ids, np.where(((theta) % np.pi < st) +
                                                   ((theta-st) % np.pi > end-st))[0])
            theta = theta[ids]
            ids_proj = np.arange(ids_proj[0], ids_proj[1])[ids]
            log.info(f'angles {theta}')
        nproj = len(theta)

        # calculate chunks
        nzchunk = int(np.ceil(nz/ncz))
        lzchunk = np.minimum(
            ncz, np.int32(nz-np.arange(nzchunk)*ncz))  # chunk sizes
        ntchunk = int(np.ceil(nproj/ncproj))
        ltchunk = np.minimum(
            ncproj, np.int32(nproj-np.arange(ntchunk)*ncproj))  # chunk sizes in proj


        tmp = literal_eval(args.nsino)
        if not isinstance(tmp, list):
            tmp = [tmp]

        params.id_slices = np.int32(np.array(tmp)*(nz*2**args.binning-1) /
                                    2**args.binning)*2**args.binning
        params.n = n
        params.nz = nz
        params.ncz = ncz
        params.nproj = nproj
        params.ncproj = ncproj
        params.center = center
        params.ni = ni
        params.nzi = nzi
        params.centeri = centeri
        params.ndark = ndark
        params.nflat = nflat
        params.ids_proj = ids_proj
        params.theta = theta
        params.nzchunk = nzchunk
        params.lzchunk = lzchunk
        params.ntchunk = ntchunk
        params.ltchunk = ltchunk
        params.dtype = args.dtype
        params.in_dtype = in_dtype
        params.st_n = st_n
        params.end_n = end_n

        # full shapes
        params.shape_data_full = (nproj, nz, ni)
        params.shape_data_fulln = (nproj, nz, n)

    def init_sizes_try(self):
        """Calculating sizes for try reconstruction by chunks"""

        if args.reconstruction_type == 'try':
            # invert shifts for calculations if centeri<ni for double_fov
            shift_array = np.arange(-args.center_search_width,
                                    args.center_search_width, args.center_search_step*2**args.binning).astype('float32')/2**args.binning
            save_centers = (params.centeri - shift_array) * \
                2**args.binning+params.st_n
            if (args.file_type == 'double_fov') and (params.centeri < params.ni//2):
                shift_array = -shift_array

        elif args.reconstruction_type == 'try_lamino':
            shift_array = np.arange(-args.lamino_search_width,
                                    args.lamino_search_width, args.lamino_search_step).astype('float32')
            save_centers = args.lamino_angle + shift_array
        # calculate chunks
        nschunk = int(np.ceil(len(shift_array)/params.ncz))
        lschunk = np.minimum(params.ncz, np.int32(
            len(shift_array)-np.arange(nschunk)*params.ncz))
        params.shift_array = shift_array
        params.save_centers = save_centers
        params.nschunk = nschunk
        params.lschunk = lschunk

    def init_sizes_lamino(self):
        """Calculating sizes for laminography reconstruction by chunks"""

        # calculate reconstruction height
        rh0 = int(np.ceil((params.nz*2**args.binning/np.cos(args.lamino_angle/180*np.pi)
                           )/2**args.binning/2))*2  # - args.lamino_start_row//2**args.binning
        if args.lamino_end_row == -1:
            rh = rh0
        else:
            rh = args.lamino_end_row//2**args.binning - \
                args.lamino_start_row//2**args.binning

        # calculate chunks
        nrchunk = int(np.ceil(rh/params.ncz))
        lrchunk = np.minimum(
            params.ncz, np.int32(rh-np.arange(nrchunk)*params.ncz))

        params.nrchunk = nrchunk
        params.lrchunk = lrchunk
        params.rh = rh
        params.lamino_angle = args.lamino_angle
        params.lamino_start_row = args.lamino_start_row//2**args.binning
        params.lamino_shift = (rh0//2-rh//2) - \
            args.lamino_start_row//2**args.binning

    def read_sizes(self):
        '''
        Read data sizes        
        Output: dictionary with fields

        nproji - number of projections
        nzi - detector height
        ni - detector width
        nflat - number of flat fields
        ndark - number of dark fields
        dtype - data type (e.g., 'uint16', 'uint8')
        '''
        sizes = {}

        with h5py.File(args.file_name) as file_in:
            data = file_in['/exchange/data']
            nproj, nzi, ni = data.shape[:]

            sizes['dtype'] = data.dtype
            sizes['nproji'] = nproj
            sizes['nzi'] = nzi
            sizes['ni'] = ni

        with h5py.File(args.flat_file_name) as file_in:
            flat = file_in['/exchange/data_white']
            nflat = flat.shape[0]
            sizes['nflat'] = nflat

        with h5py.File(args.dark_file_name) as file_in:
            dark = file_in['/exchange/data_dark']
            ndark = dark.shape[0]
            sizes['ndark'] = ndark

        return sizes

    def read_theta(self, projections):
        """Read projection angles (in radians)"""

        with h5py.File(args.file_name) as file_in:
                if '/exchange/theta' in file_in:
                        theta = file_in['/exchange/theta'][:].astype('float32') / 180 * np.pi
                else:
                        # If 'theta' doesn't exist, calculate it over the range [0, proj]
                        theta = np.linspace(0, np.pi, projections, dtype='float32')
        return theta

    def read_data_chunk_to_queue(self, data_queue, ids_proj, st_z, end_z, st_n, end_n, id_z, in_dtype):
        '''
        Read a data chunk (proj, flat,dark) from the storage to a python queue, with downsampling
        Input:

        data_queue - a python queue for synchronous read/writes
        ids_proj - the first and last projection ids for reconstruction (e.g, (0,1500)), or np.array with ids (if some angles are blocked and should be ignored),
        st_z - start row in z
        end_z - end row in z
        st_n - start column in x
        end_n - end column in x
        id_z - chunk id (for ordering after parallel processing) 
        id_dtype - input data type (e.g. uint8), or reconstruction type (if binning>0)
        '''

        with h5py.File(args.file_name) as fid:
            if isinstance(ids_proj, np.ndarray):
                # data = fid['/exchange/data'][ids_proj, st_z:end_z,
                #                             st_n:end_n].astype(in_dtype, copy=False)
                data = fid['/exchange/data'][:, st_z:end_z,
                                             st_n:end_n][ids_proj].astype(in_dtype, copy=False)
            else:
                data = fid['/exchange/data'][ids_proj[0]:ids_proj[1],
                                             st_z:end_z, st_n:end_n].astype(in_dtype, copy=False)

        with h5py.File(args.dark_file_name) as fid:
            data_dark = fid['/exchange/data_dark'][:,
                                                   st_z:end_z, st_n:end_n].astype(in_dtype, copy=False)

        with h5py.File(args.flat_file_name) as fid:
            data_flat = fid['/exchange/data_white'][:,
                                                    st_z:end_z, st_n:end_n].astype(in_dtype, copy=False)
            item = {}
            item['data'] = utils.downsample(data, args.binning)
            item['flat'] = utils.downsample(data_flat, args.binning)
            item['dark'] = utils.downsample(data_dark, args.binning)
            item['id'] = id_z
            data_queue.put(item)

        return data_queue

    def read_proj_chunk(self, data, st_proj, end_proj, st_z, end_z, st_n, end_n):
        """Read a chunk of projections with binning"""

        with h5py.File(args.file_name) as fid:
            d = fid['/exchange/data'][args.start_proj +
                                      st_proj:args.start_proj+end_proj, st_z:end_z, st_n:end_n]
            data[st_proj:end_proj] = utils.downsample(d, args.binning)

    def read_flat_dark(self, st_n, end_n):
        """Read flat and dark"""

        with h5py.File(args.dark_file_name) as fid:
            dark = fid['/exchange/data_dark'][:,
                                              args.start_row:args.end_row, st_n:end_n]
            dark = utils.downsample(dark, args.binning)

        with h5py.File(args.flat_file_name) as fid:
            flat = fid['/exchange/data_white'][:,
                                               args.start_row:args.end_row, st_n:end_n]
            flat = utils.downsample(flat, args.binning)

        return flat, dark

    def read_pairs(self, pairs, st_z, end_z, st_n, end_n):
        """Read projection pairs for automatic search of the rotation center. E.g. pairs=[0,1499] for the regular 180 deg dataset [1500,2048,2448]. """

        with h5py.File(args.file_name) as fid:
            d = fid['/exchange/data'][pairs, st_z:end_z, st_n:end_n]
            data = utils.downsample(d, args.binning)
            return data

    def read_data_try(self, data_queue, id_slice):

        st_z = id_slice
        end_z = id_slice + 2**args.binning

        self.read_data_chunk_to_queue(
            data_queue, params.ids_proj, st_z, end_z, params.st_n, params.end_n, 0, params.in_dtype)

    def read_data_to_queue(self, data_queue, read_threads):
        """Reading data from hard disk and putting it to a queue"""

        for k in range(params.nzchunk):
            st_z = args.start_row+k*params.ncz*2**args.binning
            end_z = args.start_row + \
                (k*params.ncz+params.lzchunk[k])*2**args.binning
            ithread = utils.find_free_thread(read_threads)
            read_threads[ithread].run(self.read_data_chunk_to_queue, (
                data_queue, params.ids_proj, st_z, end_z, params.st_n, params.end_n, k, params.in_dtype))

    def read_data_parallel(self, nthreads=16):
        """Reading data in parallel (good for ssd disks)"""

        flat, dark = self.read_flat_dark(params.st_n, params.end_n)
        # parallel read of projections
        data = np.zeros(params.shape_data_full, dtype=params.in_dtype)
        lchunk = int(np.ceil(data.shape[0]/nthreads))
        procs = []
        for k in range(nthreads):
            st_proj = k*lchunk
            end_proj = min((k+1)*lchunk, args.end_proj -
                           args.start_proj)
            if st_proj >= end_proj:
                continue
            read_thread = Thread(
                target=self.read_proj_chunk, args=(data, st_proj, end_proj, args.start_row, args.end_row, params.st_n, params.end_n))
            procs.append(read_thread)
            read_thread.start()
        for proc in procs:
            proc.join()

        return data, flat, dark
