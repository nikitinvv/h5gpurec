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

''' Paganin phase retrieval implementation 
'''

import cupy as cp
from cupy.fft import fft2, ifft2

__all__ = ['paganin_filter', ]

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def paganin_filter(
        data, pixel_size=1e-4, dist=50, energy=20, alpha=1e-3, method='paganin', db=1000, W=2e-4, pad=True):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter for Paganin method.
    method : string
        phase retrieval method. Standard Paganin or Generalized Paganin.
    db : float, optional
        delta/beta for generalized Paganin phase retrieval 
    W :  float
        Characteristic transverse lenght scale    	
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    Returns
    -------
    ndarray
        Approximated 3D tomographic phase data.
    """

    # New dimensions and pad value after padding.
    py, pz, val = _calc_pad(data, pixel_size, dist, energy, pad)

    # Compute the reciprocal grid.
    dx, dy, dz = data.shape
    if method == 'paganin':
        w2 = _reciprocal_grid(pixel_size, dy + 2 * py, dz + 2 * pz)
        phase_filter = cp.fft.fftshift(
            _paganin_filter_factor(energy, dist, alpha, w2))
    elif method == 'Gpaganin':
        kf = _reciprocal_gridG(pixel_size, dy + 2 * py, dz + 2 * pz)
        phase_filter = cp.fft.fftshift(
            _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W))

    prj = cp.full((dy + 2 * py, dz + 2 * pz), val, dtype=data.dtype)
    _retrieve_phase(data, phase_filter, py, pz, prj, pad)

    return data

def farago_filter(
        data, pixel_size=1e-4, dist=50, energy=20, db=1000, method='farago', pad=True):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Farago:24`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    method : string
        phase retrieval method. Standard Paganin or Generalized Paganin.
    db : float, optional
        delta/beta for generalized Farago phase retrieval 
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    Returns
    -------
    ndarray
        Approximated 3D tomographic phase data.
    """

    # New dimensions and pad value after padding.
    py, pz, val = _calc_pad(data, pixel_size, dist, energy, pad)

    # Compute the reciprocal grid.
    dx, dy, dz = data.shape
    w2 = _reciprocal_grid(pixel_size, dy + 2 * py, dz + 2 * pz)
    phase_filter = cp.fft.fftshift(
        _farago_filter_factor(energy, dist, db, w2))

    prj = cp.full((dy + 2 * py, dz + 2 * pz), val, dtype=data.dtype)
    _retrieve_phase(data, phase_filter, py, pz, prj, pad)

    return data


def _retrieve_phase(data, phase_filter, px, py, prj, pad):
    dx, dy, dz = data.shape
    num_jobs = data.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()

    for m in range(num_jobs):
        prj[px:dy + px, py:dz + py] = data[m]
        prj[:px] = prj[px]
        prj[-px:] = prj[-px-1]
        prj[:, :py] = prj[:, py][:, cp.newaxis]
        prj[:, -py:] = prj[:, -py-1][:, cp.newaxis]
        fproj = fft2(prj)
        fproj *= normalized_phase_filter
        proj = cp.real(ifft2(fproj))
        if pad:
            proj = proj[px:dy + px, py:dz + py]
        data[m] = proj


def _calc_pad(data, pixel_size, dist, energy, pad):
    """
    Calculate new dimensions and pad value after padding.

    Parameters
    ----------
    data : ndarray
        3D tomographic data.
    pixel_size : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    float
        Pad value.
    """
    dx, dy, dz = data.shape
    wavelength = _wavelength(energy)
    py, pz, val = 0, 0, 0
    if pad:
        val = _calc_pad_val(data)
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)

    return py, pz, val


def _paganin_filter_factor(energy, dist, alpha, w2):
    return 1 / (_wavelength(energy) * dist * w2 / (4 * PI) + alpha)

def _farago_filter_factor(energy, dist, db, w2):
    return 1 / (cp.cos(PI*_wavelength(energy)*dist*w2) + db*cp.sin(PI*_wavelength(energy)*dist*w2))

def _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W):
    """
        Generalized phase retrieval method
        Paganin et al 2020
        diffracting feature ~2*pixel size
    """
    aph = db*(dist*_wavelength(energy))/(4*PI)
    return 1 / (1.0 - (2*aph/(W**2))*(kf-2))


def _calc_pad_width(dim, pixel_size, wavelength, dist):
    pad_pix = cp.ceil(PI * wavelength * dist / pixel_size ** 2)
    return int((pow(2, cp.ceil(cp.log2(dim + pad_pix))) - dim) * 0.5)


def _calc_pad_val(data):
    return cp.mean((data[..., 0] + data[..., -1]) * 0.5)


def _reciprocal_grid(pixel_size, nx, ny):
    """
    Calculate reciprocal grid.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, nx)
    indy = _reciprocal_coord(pixel_size, ny)
    cp.square(indx, out=indx)
    cp.square(indy, out=indy)

    idx, idy = cp.meshgrid(indy, indx)
    return idx + idy


def _reciprocal_gridG(pixel_size, nx, ny):
    """
    Calculate reciprocal grid for Generalized Paganin method.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Considering diffracting feature ~2*pixel size
    # Sampling in reciprocal space.
    indx = cp.cos(_reciprocal_coord(pixel_size, nx)*2*PI*pixel_size)
    indy = cp.cos(_reciprocal_coord(pixel_size, ny)*2*PI*pixel_size)
    idx, idy = cp.meshgrid(indy, indx)
    return idx + idy


def _reciprocal_coord(pixel_size, num_grid):
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    n = num_grid - 1
    rc = cp.arange(-n, num_grid, 2, dtype=cp.float32)
    rc *= 0.5 / (n * pixel_size)
    return rc
    
    
def make_fresnel_window(height, width, ratio, dim):
    """
    Create a low pass window based on the Fresnel propagator.
    It is used to denoise a projection image (dim=2) or a
    sinogram image (dim=1).

    Parameters
    ----------
    height : int
        Image height
    width : int
        Image width
    ratio : float
        To define the shape of the window.
    dim : {1, 2}
        Use "1" if working on a sinogram image and "2" if working on
        a projection image.

    Returns
    -------
    array_like
        2D array.
    """
    ycenter = (height - 1) * 0.5
    xcenter = (width - 1) * 0.5
    if dim == 2:
        u = (cp.arange(width) - xcenter) / width
        v = (cp.arange(height) - ycenter) / height
        u, v = cp.meshgrid(u, v)
        window = 1.0 + ratio * (u ** 2 + v ** 2)
    else:
        u = (cp.arange(width) - xcenter) / width
        win1d = 1.0 + ratio * u ** 2
        window = cp.tile(win1d, (height, 1))
    return window


def fresnel_filter(data, ratio, dim, window=None, pad=150, apply_log=True):
    """
    Apply a low-pass filter based on the Fresnel propagator to an image
    (Ref. [1]). It can be used for improving the contrast of an image.
    It's simpler than the well-known Paganin's filter (Ref. [2]).

    Parameters
    ----------
    mat : array_like
        2D array. Projection image or sinogram image.
    ratio : float
        Define the shape of the window. Larger is more smoothing.
    dim : {1, 2}
        Use "1" if working on a sinogram image and "2" if working on
        a projection image.
    window : array_like, optional
        Window for deconvolution.
    pad : int
        Padding width.
    apply_log : bool, optional
        Apply the logarithm function to the sinogram before filtering.

    Returns
    -------
    array_like
        2D array. Filtered image.

    References
    ----------
    [1] : https://doi.org/10.1364/OE.418448

    [2] : https://tinyurl.com/2f8nv875
    """
    if apply_log:
        data = -cp.log(data)

    if dim == 2:
        (nrow, ncol, num_jobs) = data.shape
    else:    
        (nrow, num_jobs, ncol) = data.shape
    #create FF window
    if window is None:
        if dim == 2:  # On projections
            window = make_fresnel_window(nrow, ncol, ratio, dim)
        else:
            window = make_fresnel_window(nrow, ncol, ratio, dim)       

    for m in range(num_jobs):
        mat = data[:, m, :] if dim != 2 else data[:, :, m]
        if dim == 2:
            mat_pad = cp.pad(data[:, :, m], pad, mode="edge")
            win_pad = cp.pad(window, pad, mode="edge")
            #win_pad = cp.repeat(win_pad[:,:,cp.newaxis], num_jobs,axis=2)
            mat_dec = cp.fft.ifft2(cp.fft.fft2(mat_pad) / cp.fft.ifftshift(win_pad))
            mat_dec = cp.real(mat_dec[pad:pad + nrow, pad:pad + ncol])
            data[:, :, m] = mat_dec
        else:  # On sinograms
            mat_pad = cp.pad(data[:, m, :], ((0, 0), (pad, pad)), mode='edge')
            win_pad = cp.pad(window, ((0, 0), (pad, pad)), mode="edge")
            #win_pad = cp.repeat(win_pad[:,cp.newaxis,:], num_jobs,axis=1)
            mat_fft = cp.fft.fftshift(cp.fft.fft(mat_pad), axes=1) / win_pad
            mat_dec = cp.fft.ifft(cp.fft.ifftshift(mat_fft, axes=1))
            mat_dec = cp.real(mat_dec[:, pad:pad + ncol])
            data[:, m, :] = mat_dec

    if apply_log:
        data = cp.exp(-data)

    return data
    
    
