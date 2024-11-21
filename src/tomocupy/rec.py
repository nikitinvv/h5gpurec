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

from tomocupy import utils
from tomocupy import logging
from tomocupy.processing import proc_functions
from tomocupy.reconstruction import backproj_functions
from tomocupy.global_vars import args, params

from threading import Thread
from queue import Queue
import cupy as cp
import numpy as np
import signal

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['GPURec', ]


log = logging.getLogger(__name__)


class GPURec():
    '''
    Class for tomographic reconstruction on GPU with conveyor data processing by sinogram chunks (in z direction).
    Data reading/writing are done in separate threads, CUDA Streams are used to overlap CPU-GPU data transfers with computations.
    The implemented reconstruction method is Fourier-based with exponential functions for interpoaltion in the frequency domain (implemented with CUDA C).
    '''

    def __init__(self, cl_reader, cl_writer):

        # Set ^C, ^Z interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTERM, utils.signal_handler)

        # # use pinned memory
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

        # chunks for processing
        self.shape_data_chunk = (params.nproj, params.ncz, params.ni)
        self.shape_recon_chunk = (params.ncz, params.n, params.n)
        self.shape_dark_chunk = (params.ndark, params.ncz, params.ni)
        self.shape_flat_chunk = (params.nflat, params.ncz, params.ni)

        # init tomo functions
        self.cl_proc_func = proc_functions.ProcFunctions()
        self.cl_backproj_func = backproj_functions.BackprojFunctions()

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # threads for data reading from disk
        self.read_threads = []
        for k in range(args.max_read_threads):
            self.read_threads.append(utils.WRThread())

        # threads for data writing to disk
        self.write_threads = []
        for k in range(args.max_write_threads):
            self.write_threads.append(utils.WRThread())

        self.data_queue = Queue(32)

        # thread for reading data to a queue
        self.main_read_thread = Thread(
            target=cl_reader.read_data_to_queue, args=(self.data_queue, self.read_threads))

        # additional refs
        self.cl_reader = cl_reader
        self.cl_writer = cl_writer

    def recon_all(self):
        """Reconstruction of data from an h5file by splitting into sinogram chunks"""

        # start readint to the queue
        self.main_read_thread.start()

        # refs for faster access
        dtype = params.dtype
        in_dtype = params.in_dtype
        nzchunk = params.nzchunk
        lzchunk = params.lzchunk
        ncz = params.ncz

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk], dtype=in_dtype))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, *self.shape_dark_chunk], dtype=in_dtype))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, *self.shape_flat_chunk], dtype=in_dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, *self.shape_data_chunk], dtype=in_dtype)
        item_gpu['dark'] = cp.zeros(
            [2, *self.shape_dark_chunk], dtype=in_dtype)
        item_gpu['flat'] = cp.ones(
            [2, *self.shape_flat_chunk], dtype=in_dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([args.max_write_threads, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        # chunk ids with parallel read
        ids = []
        st = end = []
        log.info('Full reconstruction')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nzchunk+2):
            utils.printProgressBar(
                k, nzchunk+1, self.data_queue.qsize(), length=40)
            if (k > 0 and k < nzchunk+1):
                with self.stream2:  # reconstruction
                    data = item_gpu['data'][(k-1) % 2]
                    dark = item_gpu['dark'][(k-1) % 2]
                    flat = item_gpu['flat'][(k-1) % 2]
                    rec = rec_gpu[(k-1) % 2]
                    st = ids[k-1]*ncz+args.start_row//2**args.binning
                    end = st+lzchunk[ids[k-1]]
                    data = self.cl_proc_func.proc_sino(data, dark, flat)
                    data = self.cl_proc_func.proc_proj(data, st, end)
                    data = cp.ascontiguousarray(data.swapaxes(0, 1))
                    sht = cp.tile(np.float32(0), data.shape[0])
                    data = self.cl_backproj_func.fbp_filter_center(data, sht)
                    self.cl_backproj_func.cl_rec.backprojection(
                        rec, data, self.stream2)

            if (k > 1):
                with self.stream3:  # gpu->cpu copy
                    # find free thread
                    ithread = utils.find_free_thread(self.write_threads)
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[ithread])
            if (k < nzchunk):
                # copy to pinned memory
                item = self.data_queue.get()
                ids.append(item['id'])
                item_pinned['data'][k % 2, :, :lzchunk[ids[k]]] = item['data']
                item_pinned['dark'][k % 2, :, :lzchunk[ids[k]]] = item['dark']
                item_pinned['flat'][k % 2, :, :lzchunk[ids[k]]] = item['flat']

                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if (k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                st = ids[k-2]*ncz+args.start_row//2**args.binning
                end = st+lzchunk[ids[k-2]]
                self.write_threads[ithread].run(
                    self.cl_writer.write_data_chunk, (rec_pinned[ithread], st, end, ids[k-2], args.start_row))

            self.stream1.synchronize()
            self.stream2.synchronize()

        for t in self.write_threads:
            t.join()

    def recon_try(self):
        """GPU reconstruction of 1 slice for different centers"""

        for id_slice in params.id_slices:
            log.info(f'Processing slice {id_slice}')
            self.cl_reader.read_data_try(self.data_queue, id_slice)
            # read slice
            item = self.data_queue.get()

            # copy to gpu
            data = cp.array(item['data'])
            dark = cp.array(item['dark'])
            flat = cp.array(item['flat'])

            # preprocessing
            data = self.cl_proc_func.proc_sino(data, dark, flat)
            data = self.cl_proc_func.proc_proj(data)
            data = cp.ascontiguousarray(data.swapaxes(0, 1))

            # refs for faster access
            dtype = params.dtype
            nschunk = params.nschunk
            lschunk = params.lschunk
            ncz = params.ncz

            # pinned memory for reconstrution
            rec_pinned = utils.pinned_array(
                np.zeros([args.max_write_threads, *self.shape_recon_chunk], dtype=dtype))
            # gpu memory for reconstrution
            rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

            # Conveyor for data cpu-gpu copy and reconstruction
            for k in range(nschunk+2):
                utils.printProgressBar(
                    k, nschunk+1, self.data_queue.qsize(), length=40)
                if (k > 0 and k < nschunk+1):
                    with self.stream2:  # reconstruction
                        sht = cp.pad(cp.array(params.shift_array[(
                            k-1)*ncz:(k-1)*ncz+lschunk[k-1]]), [0, ncz-lschunk[k-1]])
                        datat = cp.tile(data, [ncz, 1, 1])
                        datat = self.cl_backproj_func.fbp_filter_center(
                            datat, sht)
                        self.cl_backproj_func.cl_rec.backprojection(
                            rec_gpu[(k-1) % 2], datat, self.stream2)
                if (k > 1):
                    with self.stream3:  # gpu->cpu copy
                        # find free thread
                        ithread = utils.find_free_thread(self.write_threads)
                        rec_gpu[(k-2) % 2].get(out=rec_pinned[ithread])
                self.stream3.synchronize()
                if (k > 1):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    for kk in range(lschunk[k-2]):
                        self.write_threads[ithread].run(self.cl_writer.write_data_try, (
                            rec_pinned[ithread, kk], params.save_centers[(k-2)*ncz+kk], id_slice))

                self.stream1.synchronize()
                self.stream2.synchronize()

            for t in self.write_threads:
                t.join()
