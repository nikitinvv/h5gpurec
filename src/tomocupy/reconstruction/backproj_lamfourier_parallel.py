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
from tomocupy.reconstruction import fbp_filter
from tomocupy.reconstruction import lamfourierrec
from tomocupy.global_vars import args, params
from threading import Thread
import cupy as cp
import numpy as np

log = logging.getLogger(__name__)


class BackprojLamFourierParallel():
    """Fourier-based method for laminography reconstruction with chunk data processing (https://arxiv.org/abs/2401.11101)    
    """

    def __init__(self, cl_writer):
        self.n0 = params.rh
        self.n1 = params.n
        self.n2 = params.n
        self.ntheta = params.nproj
        self.nthetac = params.ncproj
        self.detw = params.n
        self.deth = params.nz
        self.n1c = params.ncz
        self.dethc = params.ncz
        self.center = params.center
        self.ne = 4*self.detw

        self.cl_lamfourier = lamfourierrec.LamFourierRec(
            self.n0, self.n1, self.n2, self.ntheta, self.detw, self.deth, self.n1c, self.nthetac, self.dethc)

        ################################
        s3 = [self.ntheta, self.deth, self.detw]
        s2 = [self.ntheta, self.deth, (self.detw//2+1)]
        s1 = [self.n1, (self.deth//2+1), self.n2]
        s0 = [self.n1, self.n0, self.n2]

        s5c = [self.nthetac, self.deth, self.detw]
        s4c = [self.nthetac, self.deth, self.detw//2+1]
        s3c = [2*self.ntheta, self.dethc, self.detw//2+1]
        s2c = [self.n1, self.dethc, self.n2]
        s1c = [self.n1c, self.deth//2+1, self.n2]
        s0c = [self.n1c, self.n0, self.n2]

        # blocks to reuse memory
        global_block_size = max(np.prod(s0), np.prod(
            s1)*2, np.prod(s2)*2, np.prod(s3))
        gpu_block_size = max(np.prod(s0c), np.prod(
            s1c)*2, np.prod(s2c)*2, np.prod(s3c)*2, np.prod(s4c)*2, np.prod(s5c))

        self.pab0 = np.empty(global_block_size, dtype='float32')
        self.pab1 = np.empty(global_block_size, dtype='float32')

        self.pa33 = self.pab0[:np.prod(s3)].reshape(s3)
        self.pa22 = self.pab1[:np.prod(s2)*2].view('complex64').reshape(s2)
        self.pa11 = self.pab0[:np.prod(s1)*2].view('complex64').reshape(s1)
        self.pa00 = self.pab1[:np.prod(s0)].reshape(s0)

        self.gab0 = cp.empty(2*gpu_block_size, dtype='float32')
        self.gab1 = cp.empty(2*gpu_block_size, dtype='float32')
        self.gpab0 = utils.pinned_array(
            np.empty(gpu_block_size, dtype='float32'))
        self.gpab1 = utils.pinned_array(
            np.empty(gpu_block_size, dtype='float32'))

        self.ga55 = self.gab0[:2*np.prod(s5c)].reshape(2, *s5c)
        self.ga44 = self.gab1[:2 *
                              np.prod(s4c)*2].view('complex64').reshape(2, *s4c)
        self.ga33 = self.gab0[:2 *
                              np.prod(s3c)*2].view('complex64').reshape(2, *s3c)
        self.ga22 = self.gab1[:2 *
                              np.prod(s2c)*2].view('complex64').reshape(2, *s2c)
        self.ga11 = self.gab0[:2 *
                              np.prod(s1c)*2].view('complex64').reshape(2, *s1c)
        self.ga00 = self.gab1[:2*np.prod(s0c)].reshape(2, *s0c)

        self.gpa55 = self.gpab0[:np.prod(s5c)].reshape(s5c)
        self.gpa44 = self.gpab1[:np.prod(s4c)*2].view('complex64').reshape(s4c)
        self.gpa33 = self.gpab0[:np.prod(s3c)*2].view('complex64').reshape(s3c)
        self.gpa22 = self.gpab1[:np.prod(s2c)*2].view('complex64').reshape(s2c)
        self.gpa11 = self.gpab0[:np.prod(s1c)*2].view('complex64').reshape(s1c)
        self.gpa00 = self.gpab1[:np.prod(s0c)].reshape(s0c)

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # threads for data writing to disk
        self.write_threads = []
        for k in range(args.max_write_threads):
            self.write_threads.append(utils.WRThread())

        self.cl_filter = fbp_filter.FBPFilter(
            self.ne, self.deth, self.nthetac, args.dtype)  # note filter is applied on projections, not sinograms as in other methods

        self.wfilter = self.cl_filter.calc_filter(args.fbp_filter)

        self.cl_writer = cl_writer

        self.rec_fun = self.rec_lam

    def usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu, out_p, inp_p, phi):
        log.info("usfft1d by chunks.")
        nchunk = int(np.ceil(self.n1/self.n1c))

        for k in range(nchunk+2):
            utils.printProgressBar(
                k, nchunk+1, nchunk-k+1, length=40)
            if (k > 0 and k < nchunk+1):
                with self.stream2:  # gpu computations
                    self.cl_lamfourier.usfft1d_adj(
                        out_gpu[(k-1) % 2], inp_gpu[(k-1) % 2], phi, self.stream2)
            if (k > 1):
                with self.stream3:  # gpu->cpu pinned copy
                    # contiguous copy, fast  # not swapaxes
                    out_gpu[(k-2) % 2].get(out=out_p)

            if (k < nchunk):
                st, end = k*self.n1c, min(self.n1, (k+1)*self.n1c)
                s = end-st
                # inp_p[:s] = inp_t[st:end]
                utils.copy(inp_t[st:end], inp_p)
                with self.stream1:
                    inp_gpu[k % 2].set(inp_p)

            self.stream3.synchronize()

            if (k > 1):
                # cpu pinned->cpu copy
                st, end = (k-2)*self.n1c, min(self.n1, (k-1)*self.n1c)
                s = end-st
                # out_t[st:end] = out_p[:s]
                utils.copy(out_p[:s], out_t[st:end])

            self.stream1.synchronize()
            self.stream2.synchronize()

    def usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, out_p, inp_p, theta, phi):
        log.info("usfft2d by chunks.")

        theta = cp.array(theta)
        nchunk = int(np.ceil((self.deth//2+1)/self.dethc))
        for k in range(nchunk+2):
            utils.printProgressBar(
                k, nchunk+1, nchunk-k+1, length=40)
            if (k > 0 and k < nchunk+1):
                with self.stream2:  # gpu computations
                    self.cl_lamfourier.usfft2d_adj(
                        out_gpu[(k-1) % 2], inp_gpu[(k-1) % 2], theta, phi, k-1, self.stream2)
            if (k > 1):
                with self.stream3:  # gpu->cpu copy
                    out_gpu[(k-2) % 2].get(out=out_p)

            if (k < nchunk):
                # cpu -> cpu pinned copy
                st, end = k*self.dethc, min(self.deth//2+1, (k+1)*self.dethc)
                s = end-st
                utils.copy(inp[:, st:end], inp_p[:self.ntheta, :s])
                # copy the flipped part of the array for handling r2c FFT
                if k == 0:
                    utils.copy(inp[:, self.deth-end+1:self.deth -
                               st+1], inp_p[self.ntheta:, -s:-1])
                    utils.copy(inp[:, 0], inp_p[self.ntheta:, -1])
                else:
                    utils.copy(inp[:, self.deth-end+1:self.deth -
                               st+1], inp_p[self.ntheta:, -s:])

                with self.stream1:  # cpu pinned->gpu copy
                    inp_gpu[k % 2].set(inp_p)

            self.stream3.synchronize()
            if (k > 1):
                # cpu pinned->cpu copy
                st, end = (k-2)*self.dethc, min(self.deth //
                                                2+1, (k-1)*self.dethc)
                s = end-st
                utils.copy(out_p[:, :s], out[:, st:end])

            self.stream1.synchronize()
            self.stream2.synchronize()

    def fft2_chunks(self, out, inp, out_gpu, inp_gpu, out_p, inp_p):
        log.info("fft2 by chunks.")

        nchunk = int(np.ceil(self.ntheta/self.nthetac))
        for k in range(nchunk+2):
            utils.printProgressBar(
                k, nchunk+1, nchunk-k+1, length=40)
            if (k > 0 and k < nchunk+1):
                with self.stream2:  # gpu computations
                    data0 = inp_gpu[(k-1) % 2]
                    data0 = self.fbp_filter_center(
                        data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                    self.cl_lamfourier.fft2d_fwd(
                        out_gpu[(k-1) % 2], data0, self.stream2)
            if (k > 1):
                with self.stream3:  # gpu->cpu pinned copy
                    out_gpu[(k-2) % 2].get(out=out_p)

            if (k < nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k * \
                        self.nthetac, min(self.ntheta, (k+1)*self.nthetac)
                    s = end-st
                    utils.copy(inp[st:end], inp_p[:s])
                    inp_gpu[k % 2].set(inp_p)

            self.stream3.synchronize()
            if (k > 1):
                # cpu pinned ->cpu copy
                st, end = (k-2)*self.nthetac, min(self.ntheta,
                                                  (k-1)*self.nthetac)
                s = end-st
                utils.copy(out_p[:s], out[st:end])
            self.stream1.synchronize()
            self.stream2.synchronize()

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        tmp = cp.pad(
            data, ((0, 0), (0, 0), (self.ne//2-self.n2//2, self.ne//2-self.n2//2)), mode='edge')
        t = cp.fft.rfftfreq(self.ne).astype('float32')
        w = self.wfilter*cp.exp(-2*cp.pi*1j*t*(-self.center +
                                               sht[:, cp.newaxis]+self.n2/2))  # center fix
        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, self.ne//2-self.n2//2:self.ne//2+self.n2//2]

        return data  # reuse input memory

    def rec_lam(self, data):
        """Reconstruction via the the Fourier-based method"""

        utils.copy(data, self.pa33)
        self.fft2_chunks(self.pa22, self.pa33, self.ga44,
                         self.ga55, self.gpa44, self.gpa55)
        self.usfft2d_chunks(self.pa11, self.pa22, self.ga22, self.ga33, self.gpa22,
                            self.gpa33, params.theta, np.pi/2+params.lamino_angle/180*np.pi)
        self.usfft1d_chunks(self.pa00, self.pa11, self.ga00, self.ga11,
                            self.gpa00, self.gpa11, np.pi/2+params.lamino_angle/180*np.pi)
        u = utils.copyTransposed(self.pa00)
        self.write_parallel(u)

    def write_parallel(self, u, nthreads=16):
        nchunk = int(np.ceil(u.shape[0]/nthreads))
        mthreads = []
        for k in range(nthreads):
            st = k*nchunk+params.lamino_start_row
            end = min((k+1)*nchunk, u.shape[0])+params.lamino_start_row
            th = Thread(target=self.cl_writer.write_data_chunk, args=(
                u[k*nchunk:min((k+1)*nchunk, u.shape[0])], st, end, k, args.start_row))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
