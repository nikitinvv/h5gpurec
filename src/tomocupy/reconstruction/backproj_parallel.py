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
import cupy as cp
import numpy as np
from tomocupy.reconstruction import backproj_functions
from tomocupy.global_vars import args, params

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['BackprojParallel']

log = logging.getLogger(__name__)


class BackprojParallel():

    def __init__(self, cl_writer):

        # init tomo functions
        self.cl_backproj_func = backproj_functions.BackprojFunctions()
        # chunks for processing
        self.shape_data_chunk_zn = (params.nproj, params.ncz, params.n)
        self.shape_data_chunk_t = (params.ncproj, params.nz, params.ni)
        self.shape_data_chunk_tn = (params.ncproj, params.nz, params.n)
        self.shape_recon_chunk = (params.ncz, params.n, params.n)

        if args.reconstruction_type == 'full':
            if args.lamino_angle == 0:
                rec_fun = self.recon_sino_parallel
            else:
                rec_fun = self.recon_sino_proj_parallel
        elif args.reconstruction_type == 'try':
            if args.lamino_angle == 0:
                rec_fun = self.recon_try_sino_parallel
            else:
                rec_fun = self.recon_try_sino_proj_parallel
        elif args.reconstruction_type == 'try_lamino':
            rec_fun = self.recon_try_lamino_sino_proj_parallel

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # threads for data writing to disk
        self.write_threads = []
        for k in range(args.max_write_threads):
            self.write_threads.append(utils.WRThread())

        self.rec_fun = rec_fun
        self.cl_writer = cl_writer

    def recon_sino_proj_parallel(self, data):
        """Reconstruction by splitting into sinogram and projectionchunks"""

        # refs for faster access
        nrchunk = params.nrchunk
        lrchunk = params.lrchunk
        ncz = params.ncz
        ntchunk = params.ntchunk
        ltchunk = params.ltchunk
        ncproj = params.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=params.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros(
            [2, *self.shape_data_chunk_tn], dtype=params.dtype)
        theta_gpu = cp.array(params.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([args.max_write_threads, *self.shape_recon_chunk], dtype=params.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=params.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for kr in range(nrchunk+2):
            utils.printProgressBar(
                kr, nrchunk+1, nrchunk-kr+1, length=40)
            rec_gpu[(kr-1) % 2][:] = 0
            for kt in range(ntchunk+2):
                if (kr > 0 and kr < nrchunk+1 and kt > 0 and kt < ntchunk+1):
                    with self.stream2:  # reconstruction
                        data0 = data_gpu[(kt-1) % 2]
                        theta0 = theta_gpu[(kt-1)*ncproj:(kt-1)
                                           * ncproj+ltchunk[(kt-1)]]
                        rec = rec_gpu[(kr-1) % 2]

                        data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                        data0 = self.cl_backproj_func.fbp_filter_center(
                            data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                        self.cl_backproj_func.cl_rec.backprojection(
                            rec, data0, self.stream2, theta0, params.lamino_angle, (kr-1)*ncz+args.lamino_start_row//2**args.binning)

                if (kr > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec_gpu[(kr-2) % 2, :] = rec_gpu[(kr-2) % 2]
                        ithread = 0
                        while True:
                            if not self.write_threads[ithread].is_alive():
                                break
                            ithread = (
                                ithread+1) % args.max_write_threads
                        rec_gpu[(kr-2) % 2].get(out=rec_pinned[ithread])
                if (kt < ntchunk):
                    # copy to pinned memory
                    data_pinned[kt % 2][:ltchunk[kt]
                                        ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                    data_pinned[kt % 2][ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        data_gpu[kt % 2].set(data_pinned[kt % 2])
                self.stream3.synchronize()
                if (kr > 1 and kt == 0):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    st = (kr-2)*ncz+args.lamino_start_row//2**args.binning
                    end = st+lrchunk[kr-2]
                    self.write_threads[ithread].run(
                        self.cl_writer.write_data_chunk, (rec_pinned[ithread], st, end, kr-2,args.start_row))

                self.stream1.synchronize()
                self.stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for t in self.write_threads:
            t.join()

    def recon_try_sino_proj_parallel(self, data):
        """Reconstruction of 1 slice with different centers by splitting data into sinogram and projection chunks"""

        # refs for faster access
        nschunk = params.nschunk
        lschunk = params.lschunk
        ncz = params.ncz
        ntchunk = params.ntchunk
        ltchunk = params.ltchunk
        ncproj = params.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=params.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros(
            [2, *self.shape_data_chunk_tn], dtype=params.dtype)
        theta_gpu = cp.array(params.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([args.max_write_threads, *self.shape_recon_chunk], dtype=params.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=params.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for id_slice in params.id_slices:
            log.info(f'Processing slice {id_slice}')
            for ks in range(nschunk+2):
                rec_gpu[(ks-1) % 2][:] = 0

                for kt in range(ntchunk+2):
                    if (ks > 0 and ks < nschunk+1 and kt > 0 and kt < ntchunk+1):
                        with self.stream2:  # reconstruction
                            sht = cp.array(params.shift_array[(
                                ks-1)*ncz:(ks-1)*ncz+lschunk[ks-1]])
                            theta0 = theta_gpu[(kt-1)*ncproj:(kt-1)
                                               * ncproj+ltchunk[(kt-1)]]
                            rec = rec_gpu[(ks-1) % 2]
                            data0 = data_gpu[(kt-1) % 2]
                            data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                            data0 = self.cl_backproj_func.fbp_filter_center(
                                data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                            self.cl_backproj_func.cl_rec.backprojection_try(
                                rec, data0, sht, self.stream2, theta0, params.lamino_angle, int(id_slice//2**args.binning))

                    if (ks > 1 and kt == 0):
                        with self.stream3:  # gpu->cpu copy
                            rec_gpu[(ks-2) % 2] = rec_gpu[(ks-2) % 2]
                            # find free thread
                            ithread = utils.find_free_thread(
                                self.write_threads)
                            rec_gpu[(ks-2) % 2].get(out=rec_pinned[ithread])
                    if (kt < ntchunk):
                        # copy to pinned memory
                        data_pinned[kt % 2][:ltchunk[kt]
                                            ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                        data_pinned[kt % 2][ltchunk[kt]:] = 0
                        with self.stream1:  # cpu->gpu copy
                            data_gpu[kt % 2].set(data_pinned[kt % 2])
                    self.stream3.synchronize()
                    if (ks > 1 and kt == 0):
                        # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                        for kk in range(lschunk[ks-2]):
                            self.write_threads[ithread].run(self.cl_writer.write_data_try, (
                                rec_pinned[ithread, kk], params.save_centers[(ks-2)*ncz+kk], id_slice))

                    self.stream1.synchronize()
                    self.stream2.synchronize()
            for t in self.write_threads:
                t.join()

    def recon_try_lamino_sino_proj_parallel(self, data):
        """Reconstruction of 1 slice with different lamino angles by splitting data into sinogram and projection chunks"""

        # refs for faster access
        nschunk = params.nschunk
        lschunk = params.lschunk
        ncz = params.ncz
        ntchunk = params.ntchunk
        ltchunk = params.ltchunk
        ncproj = params.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_t], dtype=params.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_t], dtype=params.dtype)
        theta_gpu = cp.array(params.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([args.max_write_threads, *self.shape_recon_chunk], dtype=params.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=params.dtype)

        for id_slice in params.id_slices:
            log.info(f'Processing slice {id_slice}')
            # Conveyor for data cpu-gpu copy and reconstruction
            for ks in range(nschunk+2):
                rec_gpu[(ks-1) % 2][:] = 0

                for kt in range(ntchunk+2):
                    if (ks > 0 and ks < nschunk+1 and kt > 0 and kt < ntchunk+1):
                        with self.stream2:  # reconstruction
                            sht = cp.array(params.shift_array[(
                                ks-1)*ncz:(ks-1)*ncz+lschunk[ks-1]])
                            theta0 = theta_gpu[(kt-1)*ncproj:(kt-1)
                                               * ncproj+ltchunk[(kt-1)]]
                            rec = rec_gpu[(ks-1) % 2]
                            data0 = data_gpu[(kt-1) % 2]

                            data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                            data0 = self.cl_backproj_func.fbp_filter_center(
                                data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                            self.cl_backproj_func.cl_rec.backprojection_try_lamino(
                                rec, data0, sht, self.stream2, theta0, params.lamino_angle, int(id_slice//2**args.binning))

                    if (ks > 1 and kt == 0):
                        with self.stream3:  # gpu->cpu copy
                            rec_gpu[(ks-2) % 2] = rec_gpu[(ks-2) % 2]
                            # find free thread
                            ithread = utils.find_free_thread(
                                self.write_threads)
                            rec_gpu[(ks-2) % 2].get(out=rec_pinned[ithread])
                    if (kt < ntchunk):
                        # copy to pinned memory
                        data_pinned[kt % 2][:ltchunk[kt]
                                            ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                        data_pinned[kt % 2][ltchunk[kt]:] = 0
                        with self.stream1:  # cpu->gpu copy
                            data_gpu[kt % 2].set(data_pinned[kt % 2])
                    self.stream3.synchronize()
                    if (ks > 1 and kt == 0):
                        # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                        for kk in range(lschunk[ks-2]):
                            self.write_threads[ithread].run(self.cl_writer.write_data_try, (
                                rec_pinned[ithread, kk], params.save_centers[(ks-2)*ncz+kk], id_slice))
                    self.stream1.synchronize()
                    self.stream2.synchronize()
            for t in self.write_threads:
                t.join()

    def recon_sino_parallel(self, data):
        """Reconstruction by splitting into sinogram chunks"""

        # refs for faster access
        nzchunk = params.nzchunk
        lzchunk = params.lzchunk
        ncz = params.ncz

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_zn], dtype=params.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_zn], dtype=params.dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([args.max_write_threads, *self.shape_recon_chunk], dtype=params.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=params.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nzchunk+2):
            utils.printProgressBar(
                k, nzchunk+1, nzchunk-k+1, length=40)
            if (k > 0 and k < nzchunk+1):
                with self.stream2:  # reconstruction
                    data0 = data_gpu[(k-1) % 2]
                    rec = rec_gpu[(k-1) % 2]
                    data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                    data0 = self.cl_backproj_func.fbp_filter_center(
                        data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                    self.cl_backproj_func.cl_rec.backprojection(
                        rec, data0, self.stream2)
            if (k > 1):
                with self.stream3:  # gpu->cpu copy
                    # find free thread
                    ithread = utils.find_free_thread(self.write_threads)
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[ithread])

            if (k < nzchunk):
                # copy to pinned memory
                data_pinned[k % 2, :, :lzchunk[k]
                            ] = data[:, k*ncz:k*ncz+lzchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    data_gpu[k % 2].set(data_pinned[k % 2])
            self.stream3.synchronize()
            if (k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                st = (k-2)*ncz+args.start_row//2**args.binning
                end = st+lzchunk[k-2]
                self.write_threads[ithread].run(
                    self.cl_writer.write_data_chunk, (rec_pinned[ithread], st, end, k-2, args.start_row))

            self.stream1.synchronize()
            self.stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for t in self.write_threads:
            t.join()

    def recon_try_sino_parallel(self, data):
        """GPU reconstruction of 1 slice for different centers"""

        for id_slice in params.id_slices:
            log.info(f'Processing slice {id_slice}')
            data0 = data[:, id_slice//2**args.binning]
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
                    k, nschunk+1, nschunk-k+1, length=40)
                if (k > 0 and k < nschunk+1):
                    with self.stream2:  # reconstruction
                        sht = cp.pad(cp.array(params.shift_array[(
                            k-1)*ncz:(k-1)*ncz+lschunk[k-1]]), [0, ncz-lschunk[k-1]])
                        datat = cp.tile(data0, [ncz, 1, 1])
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
