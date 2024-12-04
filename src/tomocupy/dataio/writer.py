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

from tomocupy import config
from tomocupy import logging
from tomocupy.global_vars import args, params
import numpy as np
import h5py
import os
import sys
import tifffile

#Zarr writer
import shutil
import zarr
import json
from numcodecs import Blosc
import threading
import subprocess
import time

import cupy as cp
from pathlib import PosixPath
from types import SimpleNamespace


__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Writer', ]

log = logging.getLogger(__name__)


class Writer():
    '''
    Class for configuring write operations.
    '''

    def __init__(self):
        if args.reconstruction_type[:3] == 'try':
            self.init_output_files_try()
        else:
            self.init_output_files()

    def init_output_files_try(self):
        """Constructing output file names and initiating the actual files"""

        # init output files
        if (args.out_path_name is None):
            fnameout = os.path.dirname(
                args.file_name)+'_rec/try_center/'+os.path.basename(args.file_name)[:-3]
        else:
            fnameout = str(args.out_path_name)
        if not os.path.exists(fnameout):
            os.makedirs(fnameout)
        fnameout += '/recon'

        if (args.clear_folder == 'True'):
            log.info('Clearing the output folder')
            os.system(f'rm {fnameout}*')
        log.info(f'Output: {fnameout}')
        params.fnameout = fnameout

    def init_output_files(self):
        """Constructing output file names and initiating the actual files"""

        # init output files
        if (args.out_path_name is None):
            fnameout = os.path.dirname(
                args.file_name)+'_rec/'+os.path.basename(args.file_name)[:-3]+'_rec'
        else:
            fnameout = str(args.out_path_name)
        if not os.path.exists(fnameout):
            os.makedirs(fnameout)

        if (args.clear_folder == 'True'):
            log.info('Clearing the output folder')
            os.system(f'rm {fnameout}/*')

        if args.save_format == 'tiff':
            # if save results as tiff
            fnameout += '/recon'
            # saving command line for reconstruction
            fname_rec_line = os.path.dirname(fnameout)+'/rec_line.txt'
            rec_line = sys.argv
            rec_line[0] = os.path.basename(rec_line[0])
            with open(fname_rec_line, 'w') as f:
                f.write(' '.join(rec_line))

        elif args.save_format == 'h5':
            # if save results as h5 virtual datasets
            fnameout += '.h5'
            # Assemble virtual dataset
            layout = h5py.VirtualLayout(shape=(
                params.nzi/2**args.binning, params.n, params.n), dtype=params.dtype)
            if not os.path.exists(f'{fnameout[:-3]}_parts'):
                os.makedirs(f'{fnameout[:-3]}_parts')
            for k in range(params.nzchunk):
                filename = f"{fnameout[:-3]}_parts/p{k:04d}.h5"
                vsource = h5py.VirtualSource(
                    filename, "/exchange/data", shape=(params.lzchunk[k], params.n, params.n), dtype=params.dtype)
                st = args.start_row//2**args.binning+k*params.ncz
                layout[st:st+params.lzchunk[k]] = vsource

            # Add virtual dataset to output file
            rec_virtual = h5py.File(fnameout, "w")
            dset_rec = rec_virtual.create_virtual_dataset(
                "/exchange/data", layout)

            # saving command line to repeat the reconstruction as attribute of /exchange/data
            rec_line = sys.argv
            # remove full path to the file
            rec_line[0] = os.path.basename(rec_line[0])
            s = ' '.join(rec_line).encode("utf-8")
            dset_rec.attrs["command"] = np.array(
                s, dtype=h5py.string_dtype('utf-8', len(s)))
            dset_rec.attrs["axes"] = 'z:y:x'
            dset_rec.attrs["description"] = 'ReconData'
            dset_rec.attrs["units"] = 'counts'

            self.write_meta(rec_virtual)

            rec_virtual.close()
            config.update_hdf_process(
                fnameout, args, sections=config.RECON_STEPS_PARAMS)

        elif args.save_format == 'h5nolinks':
            fnameout += '.h5'
            h5w = h5py.File(fnameout, "w")
            dset_rec = h5w.create_dataset("/exchange/data", shape=(
                int(params.nzi/2**args.binning), params.n, params.n), dtype=params.dtype)

            # saving command line to repeat the reconstruction as attribute of /exchange/data
            rec_line = sys.argv
            # remove full path to the file
            rec_line[0] = os.path.basename(rec_line[0])
            s = ' '.join(rec_line).encode("utf-8")
            dset_rec.attrs["command"] = np.array(
                s, dtype=h5py.string_dtype('utf-8', len(s)))
            dset_rec.attrs["axes"] = 'z:y:x'
            dset_rec.attrs["description"] = 'ReconData'
            dset_rec.attrs["units"] = 'counts'

            self.write_meta(h5w)

            self.h5w = h5w
            self.dset_rec = dset_rec

            config.update_hdf_process(
                fnameout, args, sections=config.RECON_STEPS_PARAMS)

        elif args.save_format == 'h5sino':
            # if save results as h5 virtual datasets
            fnameout += '.h5'
            # Assemble virtual dataset
            layout = h5py.VirtualLayout(shape=(
                params.nproj, params.nzi/2**args.binning, params.n), dtype=params.dtype)
            if not os.path.exists(f'{fnameout[:-3]}_parts'):
                os.makedirs(f'{fnameout[:-3]}_parts')

            for k in range(params.nzchunk):
                filename = f"{fnameout[:-3]}_parts/p{k:04d}.h5"
                vsource = h5py.VirtualSource(
                    filename, "/exchange/data", shape=(params.nproj, params.lzchunk[k], params.n), dtype=params.dtype)
                st = args.start_row//2**args.binning+k*params.ncz
                layout[:, st:st+params.lzchunk[k]] = vsource
            # Add virtual dataset to output file
            rec_virtual = h5py.File(fnameout, "w")
            dset_rec = rec_virtual.create_virtual_dataset(
                "/exchange/data", layout)
            rec_virtual.create_dataset(
                '/exchange/theta', data=params.theta/np.pi*180)
            rec_virtual.create_dataset('/exchange/data_white', data=np.ones(
                [1, params.nzi//2**args.binning, params.n], dtype='float32'))
            rec_virtual.create_dataset('/exchange/data_dark', data=np.zeros(
                [1, params.nzi//2**args.binning, params.n], dtype='float32'))

            self.write_meta(rec_virtual)

            rec_virtual.close()
        if args.save_format == 'zarr':  # Zarr format support
            fnameout += '.zarr'
            self.zarr_output_path = fnameout
            clean_zarr(self.zarr_output_path)
            log.info(f'Zarr dataset will be created at {fnameout}')
            log.info(f"ZARR chunk structure: {args.zarr_chunk}")
              
        params.fnameout = fnameout
        log.info(f'Output: {fnameout}')
        

    def write_meta(self, rec_virtual):

        try:  # trying to copy meta
            import meta

            mp = meta.read_meta.Hdf5MetadataReader(args.file_name)
            meta_dict = mp.readMetadata()
            mp.close()
            with h5py.File(args.file_name, 'r') as f:
                log.info(
                    "  *** meta data from raw dataset %s copied to rec hdf file" % args.file_name)
                for key, value in meta_dict.items():
                    # print(key, value)
                    if key.find('exchange') != 1:
                        dset = rec_virtual.create_dataset(
                            key, data=value[0], dtype=f[key].dtype, shape=(1,))
                        if value[1] is not None:
                            s = value[1]
                            utf8_type = h5py.string_dtype('utf-8', len(s)+1)
                            dset.attrs['units'] = np.array(
                                s.encode("utf-8"), dtype=utf8_type)
        except:
            log.error('write_meta() error: Skip copying meta')
            pass


    def write_data_chunk(self, rec, st, end, k, shift_index):
        """Writing the kth data chunk to hard disk"""

        if args.save_format == 'tiff':
            for kk in range(end-st):
                fid = st+kk
                tifffile.imwrite(f'{params.fnameout}_{fid:05}.tiff', rec[kk])
        elif args.save_format == 'h5':
            filename = f"{params.fnameout[:-3]}_parts/p{k:04d}.h5"
            with h5py.File(filename, "w") as fid:
                fid.create_dataset("/exchange/data", data=rec,
                                   chunks=(1, params.n, params.n))
        elif args.save_format == 'h5nolinks':
            self.h5w['/exchange/data'][st:end, :, :] = rec[:end-st]
        elif args.save_format == 'h5sino':
            filename = f"{params.fnameout[:-3]}_parts/p{k:04d}.h5"
            with h5py.File(filename, "w") as fid:
                fid.create_dataset("/exchange/data", data=rec,
                                   chunks=(params.nproj, 1, params.n))
        elif args.save_format == 'zarr':  # Zarr format support

            chunks = [int(c.strip()) for c in args.zarr_chunk.split(',')]

            if not hasattr(self, 'zarr_array'):
                print(rec.shape)
                exit(0)
                shape = (int(params.nz / 2**args.binning), params.n, params.n)  # Full dataset shape
                print('initialize')
                print(params.nz)
                max_levels = lambda X, Y: (lambda r: (int(r).bit_length() - 1) if r != 0 else (int(X // Y).bit_length() - 1))(int(X) % int(Y))
                levels = min(max_levels(params.nz, end-st),6)
                log.info(f"Resolution levels: {levels}")
                
                scale_factors = [float(args.pixel_size) * (i + 1) for i in range(levels)]
                self.zarr_array, datasets = initialize_zarr(
                    output_path=self.zarr_output_path,
                    base_shape=shape,
                    chunks=chunks,
                    dtype=params.dtype,
                    num_levels=levels,
                    scale_factors=scale_factors,
                    compression=args.zarr_compression
                )
                fill_zarr_meta(self.zarr_array, datasets, self.zarr_output_path, args)
            # Write the current chunk to the Zarr container
            write_zarr_chunk(
                zarr_group=self.zarr_array,  # Pre-initialized Zarr container
                data_chunk=rec[:end - st],  # Data chunk to save
                start=st-shift_index,  # Starting index for this chunk along the z-axis
                end=end-shift_index    # Ending index for this chunk along the z-axis
            )

    def write_data_try(self, rec, cid, id_slice):
        """Write tiff reconstruction with a given name"""

        tifffile.imwrite(
            f'{params.fnameout}_slice{id_slice:04d}_center{cid:05.2f}.tiff', rec)
                        
            
def clean_zarr(output_path):
    if os.path.exists(output_path):
        try:
            subprocess.run(["rm", "-rf", output_path], check=True)            
            log.info(f"Successfully removed directory: {output_path}")
        except subprocess.CalledProcessError as e:
            log.error(f"Error removing directory {output_path}: {e}")
            raise
    else:
        log.warning(f"Path does not exist: {output_path}")            


def args2json(data):
    """
    Recursively convert all unsupported types (e.g., PosixPath, Namespace) to JSON-serializable types.

    Parameters:
    - data: The input data (can be dict, list, PosixPath, Namespace, etc.).

    Returns:
    - A JSON data.
    """
    if isinstance(data, PosixPath):
        return str(data)  # Convert PosixPath to string
    elif isinstance(data, SimpleNamespace):
        return {k: args2json(v) for k, v in vars(data).items()}  # Convert Namespace to dict
    elif isinstance(data, dict):
        return {k: args2json(v) for k, v in data.items()}  # Recurse into dict
    elif isinstance(data, list):
        return [args2json(item) for item in data]  # Recurse into list
    elif isinstance(data, tuple):
        return tuple(args2json(item) for item in data)  # Recurse into tuple
    else:
        return data


def fill_zarr_meta(root_group, datasets, output_path, metadata_args, mode='w'):
    """
    Fill metadata for the Zarr multiscale datasets and include additional parameters.

    Parameters:
    - root_group (zarr.Group): The root Zarr group.
    - datasets (list): List of datasets with their metadata.
    - output_path (str): Path to save the metadata file.
    - metadata_args (dict): Metadata arguments for custom configurations.
    - mode (str): Mode for metadata handling. Default is 'w'.
    """
    multiscales = [{
        "version": "0.4",
        "name": "example",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {
            "method": "skimage.transform.resize",
            "version": "0.16.1",
            "args": [True],
            "kwargs": {
                "anti_aliasing": True,
                "preserve_range": True
            }
        }
    }]

    # Update Zarr group attributes
    if mode == 'w':
        root_group.attrs.update({"multiscales": multiscales})

        # Save metadata as JSON
        metadata_file = os.path.join(output_path, 'multiscales.json')
        with open(metadata_file, 'w') as f:
            json.dump({"multiscales": multiscales}, f, indent=4)


def initialize_zarr(output_path, base_shape, chunks, dtype, num_levels, scale_factors, compression='None'):
    """
    Initialize a multiscale Zarr container with specified levels, dimensions, and compression.

    Parameters:
    - output_path (str): Path to the Zarr file.
    - base_shape (tuple): Shape of the full dataset at the highest resolution.
    - chunks (tuple): Chunk size for the dataset.
    - dtype: Data type of the dataset.
    - num_levels (int): Number of multiresolution levels.
    - scale_factors (list): List of scale factors for each level.
    - compression (str): Compression algorithm.
    
    Returns:
    - zarr.Group: The initialized Zarr group containing multiscale datasets.
    - list: Dataset metadata for multiscales.
    """
    store = zarr.DirectoryStore(output_path)
    compressor = Blosc(cname=compression, clevel=5, shuffle=2)
    root_group = zarr.group(store=store)
    
    datasets = []
    current_shape = base_shape

    for level, scale in enumerate(scale_factors):
        level_name = f"{level}"
        level_chunks = tuple(max(1, int(c) // (2 ** level)) for c in chunks)

        root_group.create_dataset(
            name=level_name,
            shape=current_shape,
            chunks=level_chunks,
            dtype=dtype,
            compressor=compressor
        )

        datasets.append({
            "path": level_name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale] * 3}
            ]
        })

        current_shape = tuple(max(1, s // 2) for s in current_shape)

    return root_group, datasets

def write_zarr_chunk(zarr_group, data_chunk, start, end):
    """
    Write a chunk of data into the Zarr container at all resolutions.

    Parameters:
    - zarr_group (zarr.Group): The initialized Zarr group containing multiscale datasets.
    - data_chunk (np.ndarray): The data chunk to write (highest resolution).
    - start (int): Start index in the first dimension (z-axis) for the highest resolution.
    - end (int): End index in the first dimension (z-axis) for the highest resolution.
    """
    for level in sorted(zarr_group.keys(), key=int):  # Process levels in order (0, 1, ...)
        zarr_array = zarr_group[level]  # Access the dataset for the current level

        # Calculate the downscaling factor for this resolution level
        scale_factor = 2 ** int(level)
        # Downsample data chunk for this resolution level
        if scale_factor > 1:
            downsampled_chunk = downsample_volume(data_chunk, scale_factor)
        else:
            downsampled_chunk = data_chunk

        level_start = start // scale_factor
        level_end = end // scale_factor

        expected_z_size = level_end - level_start
        actual_z_size = downsampled_chunk.shape[0]

        if actual_z_size != expected_z_size:
            downsampled_chunk = downsampled_chunk[:expected_z_size]

        # Write the downsampled chunk into the Zarr dataset
        zarr_array[level_start:level_end, :, :] = downsampled_chunk
        #log.info(f"Saved chunk to level {level} [{level_start}:{level_end}] with shape {downsampled_chunk.shape}")


def downsample_volume(volume, scale_factor):
    """
    Downsample a 3D volume by a given scale factor using NumPy.

    Parameters:
    - volume (numpy array): Input 3D volume (e.g., [z, y, x]).
    - scale_factor (int): Factor by which to downsample (e.g., 2 for halving).

    Returns:
    - numpy array: Downsampled volume.
    """
    if scale_factor == 1:
        return volume  # No downsampling needed for the highest resolution

    # Ensure the input dimensions are divisible by the scale factor
    if volume.shape[1] % scale_factor != 0 or volume.shape[2] % scale_factor != 0:
        raise ValueError("Volume dimensions must be divisible by the scale factor.")

    # Reshape the spatial dimensions for downsampling
    z, y, x = volume.shape
    downsampled = volume.reshape(
        z,
        y // scale_factor, scale_factor,
        x // scale_factor, scale_factor
    )

    return downsampled.mean(axis=(2, 4))
