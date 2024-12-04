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

import sys
import argparse
import configparser
import logging
import warnings
import inspect
import h5py
import numpy as np

from copy import copy
from pathlib import Path
from collections import OrderedDict

from tomocupy import utils
from tomocupy import __version__

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

log = logging.getLogger(__name__)


def default_parameter(func, param):
    """Get the default value for a function parameter.

    For a given function *func*, introspect the function and return
    the default value for the function parameter named *param*.

    Return
    ======
    default_val
      The default value for the parameter.

    Raises
    ======
    RuntimeError
      Raised if the function *func* has no default value for the
      requested parameter *param*.

    """
    # Retrieve the function parameter by introspection
    try:
        sig = inspect.signature(func)
        _param = sig.parameters[param]
    except TypeError as e:
        warnings.warn(str(e))
        log.warning(str(e))
        return None
    # Check if a default value exists
    if _param.default is _param.empty:
        # No default is listed in the function, so throw an exception
        msg = ("No default value given for parameter *{}* of callable {}."
               "".format(param, func))
        raise RuntimeError(msg)
    else:
        # Retrieve and return the parameter's default value
        return _param.default


LOGS_HOME = Path.home()/'logs'
CONFIG_FILE_NAME = Path.home()/'tomocupyon.conf'

SECTIONS = OrderedDict()


SECTIONS['general'] = {
    'config': {
        'default': CONFIG_FILE_NAME,
        'type': str,
        'help': "File name of configuration file",
        'metavar': 'FILE'},
    'logs-home': {
        'default': LOGS_HOME,
        'type': str,
        'help': "Log file directory",
        'metavar': 'FILE'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'},
    'config-update': {
        'default': False,
        'help': 'When set, the content of the config file is updated using the current params values',
        'action': 'store_true'},
}

SECTIONS['file-reading'] = {
    'file-name': {
        'default': '.',
        'type': Path,
        'help': "Name of the last used hdf file or directory containing multiple hdf files",
        'metavar': 'PATH'},
    'flat-file-name': {
        'default': None,
        'type': Path,
        'help': "Name of the hdf file containing flat data",
        'metavar': 'PATH'},
    'dark-file-name': {
        'default': None,
        'type': Path,
        'help': "Name of the hdf file containing dark data",
        'metavar': 'PATH'},
    'out-path-name': {
        'default': None,
        'type': Path,
        'help': "Path for output files",
        'metavar': 'PATH'},
    'file-type': {
        'default': 'standard',
        'type': str,
        'help': "Input file type",
        'choices': ['standard', 'double_fov']},
    'binning': {
        'type': utils.positive_int,
        'default': 0,
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': [0, 1, 2, 3]},
    'blocked-views': {
        'type': str,
        'default': 'none',
        'help': "Angle range for blocked views [st,end]. Can be a list of ranges(e.g. [[0,1.2],[3,3.14]])"},
}

SECTIONS['remove-stripe'] = {
    'remove-stripe-method': {
        'default': 'none',
        'type': str,
        'help': "Remove stripe method: none, fourier-wavelet, titarenko",
        'choices': ['none', 'fw', 'ti', 'vo-all']},
}

SECTIONS['fw'] = {
    'fw-sigma': {
        'default': 1,
        'type': float,
        'help': "Fourier-Wavelet remove stripe damping parameter"},
    'fw-filter': {
        'default': 'sym16',
        'type': str,
        'help': "Fourier-Wavelet remove stripe filter",
        'choices': ['haar', 'db5', 'sym5', 'sym16']},
    'fw-level': {
        'type': utils.positive_int,
        'default': 7,
        'help': "Fourier-Wavelet remove stripe level parameter"},
    'fw-pad': {
        'default': True,
        'help': "When set, Fourier-Wavelet remove stripe extend the size of the sinogram by padding with zeros",
        'action': 'store_true'},
}

SECTIONS['vo-all'] = {
    'vo-all-snr': {
        'default': 3,
        'type': float,
        'help': "Ratio used to locate large stripes. Greater is less sensitive."},
    'vo-all-la-size': {
        'default': 61,
        'type': utils.positive_int,
        'help': "Window size of the median filter to remove large stripes."},
    'vo-all-sm-size': {
        'type': utils.positive_int,
        'default': 21,
        'help': "Window size of the median filter to remove small-to-medium stripes."},
    'vo-all-dim': {
        'default': 1,
        'help': "Dimension of the window."},
}

SECTIONS['ti'] = {
    'ti-beta': {
        'default': 0.022,  # as in the paper
        'type': float,
        'help': "Parameter for ring removal (0,1)"},
    'ti-mask': {
        'default': 1,
        'type': float,
        'help': "Mask size for ring removal (0,1)"},
}

SECTIONS['retrieve-phase'] = {
    'retrieve-phase-method': {
        'default': 'none',
        'type': str,
        'help': "Phase retrieval correction method",
        'choices': ['none', 'paganin', 'Gpaganin', 'FourierFilter', 'farago']},
    'energy': {
        'default': 0,
        'type': float,
        'help': "X-ray energy [keV]"},
    'propagation-distance': {
        'default': 0,
        'type': float,
        'help': "Sample detector distance [mm]"},
    'retrieve-phase-alpha': {
        'default': 0,
        'type': float,
        'help': "Regularization parameter"},
    'retrieve-phase-delta-beta': {
        'default': 1500.0,
        'type': float,
        'help': "delta/beta material for Generalized Paganin & Farago"},
    'retrieve-phase-W': {
        'default': 2e-4,
        'type': float,
        'help': "Characteristic transverse length for Generalized Paganin"},
    'retrieve-phase-pad': {
        'type': utils.positive_int,
        'default': 1,
        'help': "Padding with extra slices in z for phase-retrieval filtering"},
    'FFratio': {
        'default': 100,
        'type': float,
        'help': "Shape of the Fourier Filter window"},  
    'FFdim': {
        'default': 2,
        'type': float,
        'help': "1 for sinograms, 2 for projections"},  
    'FFlog': {
        'default': 0,
        'type': int,
        'help': "0/1"},
    'FFpad': {
        'default': 150,
        'type': float,
        'help': "Padding size for the Fourier Filter"},              
}

SECTIONS['rotate-proj'] = {
    'rotate-proj-angle': {
        'default': 0,
        'type': float,
        'help': "Rotation angle for projections (counterclockwise)"},
    'rotate-proj-order': {
        'default': 1,
        'type': int,
        'help': "Interpolation spline order for rotation"},
}

SECTIONS['lamino'] = {
    'lamino-search-width': {
        'type': float,
        'default': 5.0,
        'help': "+/- center search width (pixel). "},
    'lamino-search-step': {
        'type': float,
        'default': 0.25,
        'help': "+/- center search step (pixel). "},
    'lamino-angle': {
        'default': 0,
        'type': float,
        'help': "Pitch of the stage for laminography"},
    'lamino-start-row': {
        'default': 0,
        'type': int,
        'help': "Start slice for lamino reconstruction"},
    'lamino-end-row': {
        'default': -1,
        'type': int,
        'help': "End slice for lamino reconstruction"},
}

SECTIONS['reconstruction-types'] = {
    'reconstruction-type': {
        'default': 'try',
        'type': str,
        'help': "Reconstruct full data set. ",
        'choices': ['full', 'try']},
    'reconstruction-algorithm': {
        'default': 'fourierrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['fourierrec', 'lprec', 'linerec']},
}

SECTIONS['reconstruction-steps-types'] = {
    'reconstruction-type': {
        'default': 'try',
        'type': str,
        'help': "Reconstruct full data set. ",
        'choices': ['full', 'try', 'try_lamino']},
    'reconstruction-algorithm': {
        'default': 'fourierrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['fourierrec', 'linerec']},
    'pre-processing': {
        'default': 'True',
        'type': str,
        'help': "Preprocess projections or not",
        'choices': ['True', 'False']},
}

SECTIONS['reconstruction'] = {
    'rotation-axis': {
        'default': -1.0,
        'type': float,
        'help': "Location of rotation axis"},
    'center-search-width': {
        'type': float,
        'default': 50.0,
        'help': "+/- center search width (pixel). "},
    'center-search-step': {
        'type': float,
        'default': 0.5,
        'help': "+/- center search step (pixel). "},
    'nsino': {
        'default': '0.5',
        'type': str,
        'help': 'Location of the sinogram used for slice reconstruction and find axis (0 top, 1 bottom). Can be given as a list, e.g. [0,0.9].'},
    'nsino-per-chunk': {
        'type': int,
        'default': 8,
        'help': "Number of sinograms per chunk. Use larger numbers with computers with larger memory. ", },
    'nproj-per-chunk': {
        'type': int,
        'default': 8,
        'help': "Number of projections per chunk. Use larger numbers with computers with larger memory.  ", },
    'start-row': {
        'type': int,
        'default': 0,
        'help': "Start slice"},
    'end-row': {
        'type': int,
        'default': -1,
        'help': "End slice"},
    'start-column': {
        'type': int,
        'default': 0,
        'help': "Start position in x"},
    'end-column': {
        'type': int,
        'default': -1,
        'help': "End position in x"},
    'start-proj': {
        'type': int,
        'default': 0,
        'help': "Start projection"},
    'end-proj': {
        'type': int,
        'default': -1,
        'help': "End projection"},
    'nproj-per-chunk': {
        'type': int,
        'default': 8,
        'help': "Number of sinograms per chunk. Use lower numbers with computers with lower GPU memory.", },
    'rotation-axis-auto': {
        'default': 'manual',
        'type': str,
        'help': "How to get rotation axis auto calculate ('auto'), or manually ('manual')",
        'choices': ['manual', 'auto', ]},
    'rotation-axis-pairs': {
        'default': '[0,0]',
        'type': str,
        'help': "Projection pairs to find rotation axis. Each second projection in a pair will be flipped and used to find shifts from the first element in a pair. The shifts are used to calculate the center.  Example [0,1499] for a 180 deg scan, or [0,1499,749,2249] for 360, etc.", },
    'rotation-axis-sift-threshold': {
        'default': 0.5,
        'type': float,
        'help': "SIFT threshold for rotation search.", },
    'rotation-axis-method': {
        'default': 'sift',
        'type': str,
        'help': "Method for automatic rotation search.",
        'choices': ['sift', 'vo']},
    'find-center-start-row': {
        'type': int,
        'default': 0,
        'help': "Start row to find the rotation center"},
    'find-center-end-row': {
        'type': int,
        'default': -1,
        'help': "End row to find the rotation center"},
    'fbp-filter': {
        'default': 'parzen',
        'type': str,
        'help': "Filter for FBP reconstruction",
        'choices': ['none', 'ramp', 'shepp', 'hann', 'hamming', 'parzen', 'cosine', 'cosine2']},
    'dezinger': {
        'type': int,
        'default': 0,
        'help': "Width of region for removing outliers"},
    'dezinger-threshold': {
        'type': int,
        'default': 5000,
        'help': "Threshold of grayscale above local median to be considered a zinger pixel"},
    'max-write-threads': {
        'type': int,
        'default': 8,
        'help': "Max number of threads for writing by chunks"},
    'max-read-threads': {
        'type': int,
        'default': 4,
        'help': "Max number of threads for reading by chunks"},
    'minus-log': {
        'default': 'True',
        'help': "Take -log or not"},
    'flat-linear': {
        'default': 'False',
        'help': "Interpolate flat fields for each projections, assumes the number of flat fields at the beginning of the scan is as the same as a the end."},
    'bright-ratio': {
        'type': float,
        'default': 1,
        'help': 'exposure time for flat fields divided by the exposure time of projections'},
    'pixel-size': {
        'default': 0,
        'type': float,
        'help': "Pixel size [microns]"},
}
SECTIONS['output'] ={
    'dtype': {
        'default': 'float32',
        'type': str,
        'choices': ['float32', 'float16'],
        'help': "Data type used for reconstruction. Note float16 works with power of 2 sizes.", },
    'save-format': {
        'default': 'tiff',
        'type': str,
        'help': "Output format",
        'choices': ['tiff', 'h5', 'h5sino', 'h5nolinks', 'zarr']},
    'zarr-compression': {
        'default': 'blosclz',
        'type': str,
        'help': "ZARR compression format",
        'choices': ['blosclz', 'lz4', 'zstd']},    
    'zarr-chunk': {
        'default': '8,64,64',
        'type': str,
        'help': "ZARR chunk size"},
    'clear-folder': {
        'default': 'False',
        'type': str,
        'help': "Clear output folder before reconstruction",
        'choices': ['True', 'False']}
}


SECTIONS['beam-hardening'] = {
    'beam-hardening-method': {
        'default': 'none',
        'type': str,
        'help': "Beam hardening method.",
        'choices': ['none', 'standard']},
    'source-distance': {
        'default': 36.0,
        'type': float,
        'help': 'Distance from source to scintillator in m'},
    'read-scintillator': {
        'default': False,
        'help': "When set, read scintillator properties from the HDF file",
        'action': 'store_true'},
    'read-pixel-size': {
        'default': False,
        'help': "When set, read effective pixel size from the HDF file",
        'action': 'store_true'},
    'scintillator-material': {
        'default': 'LuAG_Ce',
        'type': str,
        'help': 'Scintillator material for beam hardening'},
    'scintillator-thickness': {
        'default': 100.0,
        'type': float,
        'help': 'Scintillator thickness in microns'},
    'scintillator-density': {
        'default': 6.0,
        'type': float,
        'help': 'Density of scintillator in g/cm^3'},
    'sample-material': {
        'default': 'Fe',
        'type': str,
        'help': 'Sample material for beam hardening'},
    'sample-density': {
        'default': 1.0,
        'type': float,
        'help': 'Density of sample material in g/cm^3'},
    'filter-1-auto': {
        'default': False,
        'help': 'If True, read filter 1 from HDF meta data'},
    'filter-1-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 1 material for beam hardening'},
    'filter-1-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 1 thickness in microns'},
    'filter-1-density': {
        'default': 1.0,
        'type': float,
        'help': 'Filter 1 density in g/cm^3'},
    'filter-2-auto': {
        'default': False,
        'help': 'If True, read filter 2 from HDF meta data'},
    'filter-2-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 2 material for beam hardening'},
    'filter-2-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 2 thickness in microns'},
    'filter-2-density': {
        'default': 1.0,
        'type': float,
        'help': 'Filter 2 density in g/cm^3'},
    'filter-3-auto': {
        'default': False,
        'help': 'If True, read filter 3 from HDF meta data'},
    'filter-3-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 3 material in microns'},
    'filter-3-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 3 thickness for beam hardening'},
    'filter-3-density': {
        'default': 1.0,
        'type': float,
        'help': 'Filter 3 density in g/cm^3'},
    'calculate-source': {
        'default': 'none',
        'type': str,
        'help': "Use tabulated (none, default) or calculated source",
        'choices': ['none', 'standard']},
    'e-storage-ring': {
        'default': 7.0,
        'type': float,
        'help': "e-beam energy for BM source in GeV"},
    'b-storage-ring': {
        'default': 0.599,
        'type': float,
        'help': "Magnetic field for BM source in T"},
    'minimum-E': {
        'default': 1000,
        'type': float,
        'help': "Minimum energy to model in eV"},
    'maximum-E': {
        'default': 200000,
        'type': float,
        'help': "Maximum energy to model in eV"},
    'step-E': {
        'default': 500,
        'type': float,
        'help': "Energy step in eV"},
    'maximum-psi-urad': {
        'default': 40,
        'type': float,
        'help': "Maximum vertical angle from centerline to model in microradians"},

}


RECON_PARAMS = ('file-reading', 'remove-stripe',
                'reconstruction', 'fw', 'ti', 'vo-all', 'lamino', 'reconstruction-types', 'beam-hardening', 'output')
RECON_STEPS_PARAMS = ('file-reading', 'remove-stripe', 'reconstruction',
                      'retrieve-phase', 'fw', 'ti', 'vo-all', 'lamino', 'reconstruction-steps-types', 'rotate-proj', 'beam-hardening', 'output')

NICE_NAMES = ('General', 'File reading', 'Remove stripe',
              'Remove stripe FW', 'Remove stripe Titarenko', 'Remove stripe Vo' 'Retrieve phase', 'Reconstruction')


def get_config_name():
    """Get the command line --config option."""
    name = CONFIG_FILE_NAME
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--config'):
            if arg == '--config':
                return sys.argv[i + 1]
            else:
                name = sys.argv[i].split('--config')[1]
                if name[0] == '=':
                    name = name[1:]
                return name

    return name


def parse_known_args(parser, subparser=False):
    """
    Parse arguments from file and then override by the ones specified on the
    command line. Use *parser* for parsing and is *subparser* is True take into
    account that there is a value on the command line specifying the subparser.
    """
    if len(sys.argv) > 1:
        subparser_value = [sys.argv[1]] if subparser else []
        config_values = config_to_list(config_name=get_config_name())
        values = subparser_value + config_values + sys.argv[1:]
    else:
        values = ""

    return parser.parse_known_args(values)[0]


def config_to_list(config_name=CONFIG_FILE_NAME):
    """
    Read arguments from config file and convert them to a list of keys and
    values as sys.argv does when they are specified on the command line.
    *config_name* is the file name of the config file.
    """
    result = []
    config = configparser.ConfigParser()

    if not config.read([config_name]):
        return []

    for section in SECTIONS:
        for name, opts in ((n, o) for n, o in SECTIONS[section].items() if config.has_option(section, n)):
            value = config.get(section, name)

            if value != '' and value != 'None':
                action = opts.get('action', None)

                if action == 'store_true' and value == 'True':
                    # Only the key is on the command line for this action
                    result.append('--{}'.format(name))

                if not action == 'store_true':
                    if opts.get('nargs', None) == '+':
                        result.append('--{}'.format(name))
                        result.extend((v.strip() for v in value.split(',')))
                    else:
                        result.append('--{}={}'.format(name, value))

    return result


class Params(object):
    def __init__(self, sections=()):
        self.sections = sections + ('general', )

    def add_parser_args(self, parser):
        for section in self.sections:
            for name in sorted(SECTIONS[section]):
                opts = SECTIONS[section][name]
                parser.add_argument('--{}'.format(name), **opts)

    def add_arguments(self, parser):
        self.add_parser_args(parser)
        return parser

    def get_defaults(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)

        return parser.parse_args('')


def write(config_file, args=None, sections=None):
    """
    Write *config_file* with values from *args* if they are specified,
    otherwise use the defaults. If *sections* are specified, write values from
    *args* only to those sections, use the defaults on the remaining ones.
    """
    config = configparser.ConfigParser()

    for section in SECTIONS:
        config.add_section(section)
        for name, opts in SECTIONS[section].items():
            if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                value = getattr(args, name.replace('-', '_'))

                if isinstance(value, list):
                    value = ', '.join(value)
            else:
                value = opts['default'] if opts['default'] is not None else ''

            prefix = '# ' if value == '' else ''

            if name != 'config':
                config.set(section, prefix + name, str(value))

    with open(config_file, 'w') as f:
        config.write(f)


def show_config(args):
    """Log all values set in the args namespace.
    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    log.warning('tomocupy status start')
    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted(
            (k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))
        if entries:
            for entry in entries:
                value = args[entry] if args[entry] != None else "-"
                log.info("  {:<16} {}".format(entry, value))

    log.warning('tomocupy status end')


def log_values(args):
    """Log all values set in the args namespace.
    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    log.warning('tomocupyon status start')
    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted(
            (k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))

        # print('log_values', section, name, entries)
        if entries:
            log.info(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                if (value == 'none'):
                    log.warning("  {:<16} {}".format(entry, value))
                elif (value is not False):
                    log.info("  {:<16} {}".format(entry, value))
                elif (value is False):
                    log.warning("  {:<16} {}".format(entry, value))

    log.warning('tomocupyon status end')


def update_hdf_process(fname, args=None, sections=None):
    """
    Write in the hdf raw data file the content of *config_file* with values from *args* 
    if they are specified, otherwise use the defaults. If *sections* are specified, 
    write values from *args* only to those sections, use the defaults on the remaining ones.
    """
    if (args == None):
        log.warning("  *** Not saving log data to the HDF file.")

    else:
        with h5py.File(fname, 'r+') as hdf_file:
            # If the group we will write to already exists, remove it
            if hdf_file.get('/process/tomocupy-' + __version__):
                del (hdf_file['/process/tomocupy-' + __version__])
            # dt = h5py.string_dtype(encoding='ascii')
            log.info("  *** tomopy.conf parameter written to /process%s in file %s " %
                     (__version__, fname))
            config = configparser.ConfigParser()
            for section in SECTIONS:
                config.add_section(section)
                for name, opts in SECTIONS[section].items():
                    if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                        value = getattr(args, name.replace('-', '_'))
                        if isinstance(value, list):
                            # print(type(value), value)
                            value = ', '.join(value)
                    else:
                        value = opts['default'] if opts['default'] is not None else ''

                    prefix = '# ' if value == '' else ''

                    if name != 'config':
                        dataset = '/process' + '/tomocupy-' + \
                            __version__ + '/' + section + '/' + name
                        dset_length = len(str(value)) * \
                            2 if len(str(value)) > 5 else 10
                        dt = 'S{0:d}'.format(dset_length)
                        hdf_file.require_dataset(dataset, shape=(1,), dtype=dt)
                        log.info(name + ': ' + str(value))
                        try:
                            hdf_file[dataset][0] = np.bytes_(str(value))
                        except TypeError:
                            log.error(
                                "Could not convert value {}".format(value))
                            raise
