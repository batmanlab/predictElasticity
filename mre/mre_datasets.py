#! usr/bin/env python
import os
from pathlib import Path
import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
import glob
from datetime import datetime
from scipy import ndimage as ndi
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology
from skimage.filters import sobel
import pdb
from tqdm import tqdm_notebook

from mre.registration_v2 import RegPatient, Register


class MREtoXr:
    '''Make an xr dataset for mre nifti data.  If some patients are missing data, that data is added
    as a 0'd vector.  Includes two coordinate systems (one for MRI data, one for MRE data).
    Includes indicators for data quality.
    '''
    def __init__(self, data_dir, sequences, **kwargs):

        self.sequences = sequences
        self.patients = [p.stem for p in data_dir.iterdir()]

        self.nx = kwargs.get('nx', 256)
        self.ny = kwargs.get('ny', 256)
        self.nz_mri = kwargs.get('nz_mri', 32)
        self.nz_mre = kwargs.get('nz_mre', 4)
        self.mask_types = kwargs.get('mask_types', ('liver', 'mre'))
        self.primary_input = kwargs.get('primary_input', 't1_pre_water')
        self.mre_types = kwargs.get('mre_types', ('mre', 'mre_conf', 'mre_raw', 'mre_wave'))

        # Initialize empty ds
        self.init_new_ds()

    def init_new_ds(self):
        '''Initialize a new xarray dataset based on the size and shape of our inputs.'''

        if len(self.patients) == 0 or len(self.sequences) == 0:
            self.ds = None
        else:
            # Make an xarray dataset for the MRI sequences (and their masks), and the MRE sequences
            # (and their masks).  All 4 data vars are 5D (subject, sequence/mask_type, x, y, z).

            self.ds = xr.Dataset(
                {'image_mri': (['subject', 'sequence', 'x', 'y', 'z_mri'],
                               np.zeros((len(self.patients), len(self.sequences), self.nx, self.ny,
                                         self.nz_mri), dtype=np.int16)),
                 'mask_mri': (['subject', 'mask_type', 'x', 'y', 'z_mri'],
                              np.zeros((len(self.patients), len(self.mask_types), self.nx, self.ny,
                                        self.nz_mri), dtype=np.int16)),
                 'image_mre': (['subject', 'mre_type', 'x', 'y', 'z_mre'],
                               np.zeros((len(self.patients), len(self.mre_types), self.nx, self.ny,
                                         self.nz_mre), dtype=np.int16)),
                 'mask_mre': (['subject', 'mask_type', 'x', 'y', 'z_mre'],
                              np.zeros((len(self.patients), len(self.mask_types), self.nx, self.ny,
                                        self.nz_mre), dtype=np.int16)),
                 },

                coords={'subject': self.patients,
                        'sequence': self.sequences,
                        'mask_type': self.mask_types,
                        'x': range(self.nx),
                        'y': range(self.ny)[::-1],
                        'z_mri': range(self.nz_mri),
                        'z_mre': range(self.nz_mre)
                        }
            )

    def load_xr(self):
        for i, pat in enumerate(tqdm_notebook(self.patients, desc='Patients')):

            reg_pat = RegPatient(pat, self.data_dir)
            if self.primary_input not in reg_pat.images.keys():
                continue

            # Register, resize and enter into xarray for all sequences except the primary.
            # We do not keep any of their image metadata after this loop
            for seq in self.sequences:
                if seq == self.primary_input:
                    continue
                elif seq not in reg_pat.images.keys():
                    continue

                reg = Register(reg_pat.images[self.primary_input], reg_pat.images[seq])
                resized_image = self.resize_image(reg.moving_img_result, 'input_mri')

                self.ds['image_mri'].loc[{'subject': pat, 'sequence': seq}] = (
                    sitk.GetArrayFromImage(resized_image).T)

            resized_primary = self.resize_image(reg_pat.images[self.primary_input], 'input_mri')
            liver_mask = self.gen_liver_mask(resized_primary)

            self.ds['mask_mri'].loc[{'subject': pat, 'sequence': seq}] = (
                sitk.GetArrayFromImage(resized_image).T)


        # return ds
        if ds is not None:
            print(f'Writing file disk...')
            output_name = Path(data_dir.parents[0], f'xarray_{output_name}.nc')
            ds.to_netcdf(output_name)
            return ds

    def resize_image(self, input_image, var_name):
        '''Take an input image and resize it to the appropriate resolution.'''
        # Get initial and resizing params
        init_size = input_image.GetSize()
        init_spacing = input_image.GetSpacing()
        nx = self.nx
        ny = self.ny
        x_change = self.nx/init_size[0]
        y_change = self.ny/init_size[1]

        # Get variable-dependent params
        if 'mri' in var_name:
            nz = self.nz_mri
            z_change = nz/init_size[2]
        else:
            nz = self.nz_mre
            z_change = nz/init_size[2]
        if 'mask' in var_name:
            interp_method = sitk.sitkNearestNeighbor
        else:
            interp_method = sitk.sitkLinear

        # Resize image
        ref_image = sitk.GetImageFromArray(np.ones((nz, ny, nx), dtype=np.uint16))
        ref_image.SetSpacing((init_spacing[0]*x_change, init_spacing[1]*y_change,
                              init_spacing[2]*z_change))
        ref_image.SetOrigin(input_image.GetOrigin())

        center = sitk.CenteredTransformInitializer(
            ref_image, input_image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        return sitk.Resample(input_image, ref_image, center, interp_method)
