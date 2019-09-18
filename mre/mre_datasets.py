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

        self.nx = kwargs.get('nx', 256)
        self.ny = kwargs.get('ny', 256)
        self.nz_mri = kwargs.get('nz_mri', 32)
        self.nz_mre = kwargs.get('nz_mre', 4)

        self.sequences = sequences
        self.patients = [p.stem for p in data_dir.iterdir()]

        # Initialize empty ds
        self.init_new_ds()

    def init_new_ds(self):
        if len(self.patients) == 0 or len(self.sequences) == 0:
            self.ds = None
        else:
            self.ds = xr.Dataset(
                {'image_mri': (['subject', 'sequence', 'x', 'y', 'z_mri'],
                               np.zeros((len(self.patients), len(self.sequences), self.nx, self.ny,
                                         self.nz_mri), dtype=np.int16)),
                 'mask_mri': (['subject', 'x', 'y', 'z_mri'],
                              np.zeros((len(self.patients), self.nx, self.ny,
                                        self.nz_mri), dtype=np.int16)),
                 'image_mre': (['subject', 'sequence', 'x', 'y', 'z_mre'],
                               np.zeros((len(self.patients), len(self.sequences), self.nx, self.ny,
                                         self.nz_mre), dtype=np.int16)),
                 'mask_mre': (['subject', 'x', 'y', 'z_mre'],
                              np.zeros((len(self.patients), self.nx, self.ny,
                                        self.nz_mre), dtype=np.int16)),
                 },

                coords={'subject': self.patients,
                        'sequence': self.sequences,
                        'x': range(self.nx),
                        'y': range(self.ny)[::-1],
                        'z_mri': range(self.nz_mri),
                        'z_mre': range(self.nz_mre)
                        }
            )

    def load_xr(self):
        for i, pat in enumerate(tqdm_notebook(self.patients, desc='Patients')):

            reg_pat = RegPatient(pat, data_dir)
            if 't1_pre_water' not in reg_pat.images.keys():
                self.remove_patient(pat)

            # register then resize
            for seq in self.sequences:
                if seq == 't1_pre_water':
                    continue
                elif seq not in reg_pat.images.keys():
                    continue

                reg = Register(reg_pat.images['t1_pre_water'], reg_pat.images[seq])





            t1_in = get_image_match(img_files, 't1_pre_in_MR', pat, nx, ny, nz)
            t1_out = get_image_match(img_files, 't1_pre_out_MR', pat, nx, ny, nz)
            t2 = get_image_match(img_files, 't2_MR', pat, nx, ny, nz)
            t1_in_mask = get_image_match(img_files, 't1_pre_in_mask', pat, nx, ny, nz)
            t1_out_mask = get_image_match(img_files, 't1_pre_out_mask', pat, nx, ny, nz)
            t2_mask = get_image_match(img_files, 't2_mask', pat, nx, ny, nz)

            ds['image'].loc[{'subject': pat, 'sequence': 't1_in'}] = (
                sitk.GetArrayFromImage(t1_in).T)
            ds['image'].loc[{'subject': pat, 'sequence': 't1_out'}] = (
                sitk.GetArrayFromImage(t1_out).T)
            ds['image'].loc[{'subject': pat, 'sequence': 't2'}] = (
                sitk.GetArrayFromImage(t2).T)

            ds['mask'].loc[{'subject': pat, 'sequence': 't1_in'}] = (
                sitk.GetArrayFromImage(t1_in_mask).T)
            ds['mask'].loc[{'subject': pat, 'sequence': 't1_out'}] = (
                sitk.GetArrayFromImage(t1_out_mask).T)
            ds['mask'].loc[{'subject': pat, 'sequence': 't2'}] = (
                sitk.GetArrayFromImage(t2_mask).T)

        # return ds
        if ds is not None:
            print(f'Writing file disk...')
            output_name = Path(data_dir.parents[0], f'xarray_{output_name}.nc')
            ds.to_netcdf(output_name)
            return ds

