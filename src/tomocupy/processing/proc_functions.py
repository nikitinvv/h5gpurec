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

from tomocupy.processing import retrieve_phase, remove_stripe
import cupyx.scipy.ndimage as ndimage
from tomocupy.global_vars import args, params
import cupy as cp


class ProcFunctions():
    def __init__(self):

        # External processing methods initialization
        if args.beam_hardening_method != 'none':
            from tomocupy.processing.external import hardening
            self.cl_hardening = hardening.Beam_Corrector(args)

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = dark.astype(args.dtype, copy=False)
        flat0 = flat.astype(args.dtype, copy=False)
        flat0 /= args.bright_ratio  # == exposure_flat/exposure_proj
        # works only for processing all angles
        if args.flat_linear == 'True' and data.shape[0] == params.nproj:
            flat0_p0 = cp.mean(flat0[:flat0.shape[0]//2], axis=0)
            flat0_p1 = cp.mean(flat0[flat0.shape[0]//2+1:], axis=0)
            v = cp.linspace(0, 1, params.nproj)[..., cp.newaxis, cp.newaxis]
            flat0 = (1-v)*flat0_p0+v*flat0_p1
        else:
            flat0 = cp.mean(flat0, axis=0)
        dark0 = cp.mean(dark0, axis=0)
        res = (data.astype(args.dtype, copy=False)-dark0) / \
            (flat0-dark0+flat0*1e-5)

        return res

    def minus_log(self, data):
        """Taking negative logarithm"""

        data[data <= 0] = 1
        data[:] = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data  # reuse input memory

    def beamhardening(self, data, start_row, end_row):
        """Beam hardening correction"""
        if start_row == None:
            start_row = 0
        if end_row == None:
            end_row = data.shape[-2]
        current_rows = list(range(start_row, end_row))
        data[:] = self.cl_hardening.correct_centerline(data)
        data[:] = self.cl_hardening.correct_angle(data, current_rows)
        return data

    def remove_outliers(self, data):
        """Remove outliers"""

        if (int(args.dezinger) > 0):
            w = int(args.dezinger)
            if len(data.shape) == 3:
                fdata = ndimage.median_filter(data, [w, 1, w])
            else:
                fdata = ndimage.median_filter(data, [w, w])
            data[:] = cp.where(cp.logical_and(
                data > fdata, (data - fdata) > args.dezinger_threshold), fdata, data)
        return data

    def pad360(self, data):
        """Pad data with 0 to handle 360 degrees scan"""

        if (params.centeri < params.ni//2):
            # if rotation center is on the left side of the ROI
            data[:] = data[:, :, ::-1]
        w = max(1, int(2*(params.ni-params.center)))

        # smooth transition at the border
        v = cp.linspace(1, 0, w, endpoint=False)
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)
        data[:, :, -w:] *= v

        # double sinogram size with adding 0
        data = cp.pad(data, ((0, 0), (0, 0), (0, data.shape[-1])), 'constant')
        return data

    def rotate_proj(self, data, angle, order=2):
        """
        Rotate projections (fixing the roll issue)

        data: input projection data
        angle: rotary angle
        order: interpolation order for rotation
        """

        data[:] = ndimage.rotate(data, float(
            angle), reshape=False, order=order, axes=(2, 1), mode='nearest')

        return data

    def proc_sino(self, data, dark, flat, res=None):
        """Processing a sinogram data chunk"""

        if not isinstance(res, cp.ndarray):
            res = cp.zeros(data.shape, args.dtype)
        # dark flat field correrction
        data[:] = self.remove_outliers(data)
        dark[:] = self.remove_outliers(dark)
        flat[:] = self.remove_outliers(flat)
        res[:] = self.darkflat_correction(data, dark, flat)
        # remove stripes
        if args.remove_stripe_method == 'fw':
            res[:] = remove_stripe.remove_stripe_fw(
                res, args.fw_sigma, args.fw_filter, args.fw_level)
        elif args.remove_stripe_method == 'ti':
            res[:] = remove_stripe.remove_stripe_ti(
                res, args.ti_beta, args.ti_mask)
        elif args.remove_stripe_method == 'vo-all':
            res[:] = remove_stripe.remove_all_stripe(
                res, args.vo_all_snr, args.vo_all_la_size, args.vo_all_sm_size, args.vo_all_dim)

        return res

    def proc_proj(self, data, st=None, end=None, res=None):
        """Processing a projection data chunk"""

        if not isinstance(res, cp.ndarray):
            res = cp.zeros(
                [data.shape[0], data.shape[1], params.n], args.dtype)
        # retrieve phase
        if args.retrieve_phase_method == 'Gpaganin' or args.retrieve_phase_method == 'paganin':
            data[:] = retrieve_phase.paganin_filter(
                data,  args.pixel_size*1e-4, args.propagation_distance/10, args.energy,
                args.retrieve_phase_alpha, args.retrieve_phase_method, args.retrieve_phase_delta_beta,
                args.retrieve_phase_W*1e-4)  # units adjusted based on the tomopy implementation
        if args.retrieve_phase_method == 'FF':        
                data[:] = retrieve_phase.fresnel_filter(
                data, args.FFratio, args.FFdim, window=None, pad=args.FFpad, apply_log=args.FFlog)        
        if args.retrieve_phase_method == 'farago':        
                data[:] = retrieve_phase.farago_filter(
                data,  args.pixel_size*1e-4, args.propagation_distance/10, args.energy,
                args.retrieve_phase_delta_beta, args.retrieve_phase_method)
                
        if args.rotate_proj_angle != 0:
            data[:] = self.rotate_proj(
                data, args.rotate_proj_angle, args.rotate_proj_order)
        # minus log
        if args.minus_log == 'True':
            data[:] = self.minus_log(data)
        # beam hardening correction
        if args.beam_hardening_method != 'none':
            data[:] = self.beamhardening(data, st, end)
        # padding for 360 deg recon
        if args.file_type == 'double_fov':
            res[:] = self.pad360(data)
        else:
            res[:] = data[:]

        return res
