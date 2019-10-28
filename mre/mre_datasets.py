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
from scipy.signal import find_peaks
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology
from skimage.filters import sobel
import pdb
from tqdm import tqdm_notebook
# import matplotlib.pyplot as plt
# import holoviews as hv

import torch

from mre.registration_v2 import RegPatient, Register
from mre.pytorch_arch import GeneralUNet3D


class MREtoXr:
    '''Make an xr dataset for mre nifti data.  If some patients are missing data, that data is added
    as a 0'd vector.  Includes two coordinate systems (one for MRI data, one for MRE data).
    Includes indicators for data quality. Processes single patient at a time.
    '''
    def __init__(self, data_dir=None, sequences=None, patient=None, from_file=None, **kwargs):

        self.is_ipython = self._check_ipython()
        if None is data_dir is sequences is from_file:
            raise ValueError(
                '(data_dir and sequences) or (from_file) must be specified to initialize')

        if from_file:
            print(from_file)
            if '*' in from_file:
                self.ds = xr.open_mfdataset(from_file)
            else:
                self.ds = xr.open_dataset(from_file)
            return None

        elif data_dir:
            self.data_dir = data_dir
            if sequences is None:
                sequences = ['t1_pre_water', 't1_pre_in', 't1_pre_out', 't1_pre_fat', 't2']
            self.sequences = sequences
            if len(self.sequences) == 0:
                raise ValueError('No sequences specificed')
            if patient is None:
                self.patient = '0006'
            else:
                self.patient = str(patient)
            print(self.data_dir)
            print(self.patient)
            print(self.sequences)
        else:
            raise ValueError('__init__ error')

        # Load the extra args
        self.nx = kwargs.get('nx', 256)
        self.ny = kwargs.get('ny', 256)
        self.nz_mri = kwargs.get('nz_mri', 32)
        self.nz_mre = kwargs.get('nz_mre', 4)
        self.mask_types = kwargs.get('mask_types', ['liver', 'mre', 'combo'])
        self.primary_input = kwargs.get('primary_input', 't1_pre_water')
        self.mre_types = kwargs.get('mre_types', ['mre', 'mre_mask', 'mre_raw', 'mre_wave',
                                                  'mre_pred'])
        # self.mre_types = kwargs.get('mre_types', ['mre'])
        self.output_name = kwargs.get('output_name', 'test')
        self.write_file = kwargs.get('write_file', True)

        # Load the liver mask model (hard-coded for now)
        model_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/',
                          'trained_models', '01', 'model_2019-10-15_11-41-56.pkl')
        self.model = GeneralUNet3D(5, 1, 8, 1, True, False, False)
        model_dict = torch.load(model_path, map_location='cpu')
        model_dict = OrderedDict([(key[7:], val) for key, val in model_dict.items()])
        self.model.load_state_dict(model_dict, strict=True)
        self.model.eval()

        # Initialize empty ds
        self.init_new_ds()

    def get_ds(self):
        '''Return the ds loaded via 'from_file'.'''
        return self.ds

    def init_new_ds(self):
        '''Initialize a new xarray dataset based on the size and shape of our inputs.'''

        if len(self.sequences) == 0:
            raise ValueError('No sequences specificed')
        else:
            # Make an xarray dataset for the MRI sequences (and their masks), and the MRE sequences
            # (and their masks).  All 4 data vars are 5D (subject, sequence/mask_type, x, y, z).

            self.ds = xr.Dataset(
                {'image_mri': (['subject', 'sequence', 'x', 'y', 'z_mri'],
                               np.zeros((1, len(self.sequences), self.nx, self.ny,
                                         self.nz_mri), dtype=np.int16)),
                 'mask_mri': (['subject', 'mask_type', 'x', 'y', 'z_mri'],
                              np.zeros((1, len(self.mask_types), self.nx, self.ny,
                                        self.nz_mri), dtype=np.int16)),
                 'image_mre': (['subject', 'mre_type', 'x', 'y', 'z_mre'],
                               np.zeros((1, len(self.mre_types), self.nx, self.ny,
                                         self.nz_mre), dtype=np.int16)),
                 'mask_mre': (['subject', 'mask_type', 'x', 'y', 'z_mre'],
                              np.zeros((1, len(self.mask_types), self.nx, self.ny,
                                        self.nz_mre), dtype=np.int16)),
                 'mri_to_mre_idx': (['subject', 'z_mre'],
                                    np.zeros((1, self.nz_mre), dtype=np.int16)),
                 'mutual_info': (['subject', 'sequence', 'z_mre'],
                                 np.zeros((1, len(self.sequences), self.nz_mre), dtype=np.int16))
                 },

                coords={'subject': [self.patient],
                        'sequence': self.sequences,
                        'mask_type': self.mask_types,
                        'mre_type': self.mre_types,
                        'x': range(self.nx),
                        'y': range(self.ny)[::-1],
                        'z_mri': range(self.nz_mri),
                        'z_mre': range(self.nz_mre)
                        }
            )

    def load_xr(self):
        # Grab all available niftis using the RegPatient Class
        reg_pat = RegPatient(self.patient, self.data_dir)
        # Make sure patient has needed niftis:
        for seq in self.sequences:
            if seq not in reg_pat.images.keys():
                raise ValueError(f'{seq}.nii not found')
        for mre in ['mre', 'mre_raw', 'mre_mask']:
            if mre not in reg_pat.images.keys():
                raise ValueError(f'{mre}.nii not found')

        # Register, resize and load into xarray for all input image sequences except 'primary'.
        # We do not keep any of their image metadata after this loop.
        # Also, ID the correct z-locations for the mre_raw images
        resized_t1_pre, resized_primary = self.reg_inputs(reg_pat)
        self.ds['mri_to_mre_idx'].loc[dict()] = self.mri_to_mre_idx

        # Get the liver mask via the deep liver segmenter
        liver_input = self.ds['image_mri'].loc[{'subject': self.patient, 'sequence': 't1_pre_out'}]
        liver_input = liver_input.transpose('z_mri', 'y', 'x').values
        liver_mask = self.gen_liver_mask(liver_input)
        self.ds['mask_mri'].loc[{'mask_type': 'liver'}] = liver_mask

        # Add in the MRE images next.  They must be resized to the appropriate scale to match
        # the input sequences.
        # Register mre_raw to center and resize. Must be done slice by slice in 2D

        mre_raw = reg_pat.images['mre_raw']
        for i in range(mre_raw.GetSize()[2]):
            mre_raw_slice = mre_raw[:, :, i]
            np_tmp = sitk.GetArrayFromImage(mre_raw[:, :, i])
            mov_min = float(np_tmp.min())
            mov_max = float(np_tmp.max())

            resized_t1_pre_slice = resized_t1_pre[:, :, int(self.mri_to_mre_idx[i])]

            # resized_t1_pre.CopyInformation(mre_raw)

            reg = Register(resized_t1_pre_slice, mre_raw_slice, config='mre_reg')
            mre_raw_slice = sitk.RescaleIntensity(reg.moving_img_result, mov_min, mov_max)
            self.ds['image_mre'].loc[
                {'mre_type': 'mre_raw', 'z_mre': i}] = (
                    sitk.GetArrayFromImage(mre_raw_slice).T)

            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(
                reg.elastixImageFilter.GetTransformParameterMap())

            for mre_type in self.mre_types:
                if mre_type == 'mre_raw':
                    continue
                elif mre_type not in reg_pat.images.keys():
                    continue

                transformixImageFilter.SetMovingImage(reg_pat.images[mre_type][:, :, i])
                transformixImageFilter.Execute()
                mre_output = transformixImageFilter.GetResultImage()

                self.ds['image_mre'].loc[
                    {'mre_type': mre_type, 'z_mre': i}] = (
                        sitk.GetArrayFromImage(mre_output).T)

        # resized_mre = self.respace_image(reg_pat.images[mre_type], 'input_mre',
        #                                  new_spacing[0], new_spacing[1])

        # Add in the liver seg mask for MRE:
        self.ds['mask_mre'].loc[{'mask_type': 'liver'}] = (
            self.ds['mask_mri'].loc[{'mask_type': 'liver', 'z_mri': self.mri_to_mre_idx}])

        # Add in the mre mask based on the conf image:
        self.gen_elast_mask(self.patient)
        self.ds['mask_mri'].loc[
            {'mask_type': 'mre', 'z_mri': self.mri_to_mre_idx}] = (
                self.ds['mask_mre'].loc[{'mask_type': 'mre'}])

        # Add in the combo masks:
        combo_mri = (self.ds['mask_mri'].sel(subject=self.patient, mask_type='mre').values *
                     self.ds['mask_mri'].sel(subject=self.patient, mask_type='liver').values)
        # combo_mri = morphology.binary_erosion(combo_mri.values)
        # combo_mri = morphology.binary_erosion(combo_mri)
        combo_mri *= morphology.remove_small_objects(combo_mri.astype(bool))
        self.ds['mask_mri'].loc[{'mask_type': 'combo'}] = combo_mri

        combo_mre = (self.ds['mask_mre'].sel(subject=self.patient, mask_type='mre').values *
                     self.ds['mask_mre'].sel(subject=self.patient, mask_type='liver').values)
        # combo_mre = morphology.binary_erosion(combo_mre.values)
        # combo_mre = morphology.binary_erosion(combo_mre)
        combo_mre *= morphology.remove_small_objects(combo_mre.astype(bool))
        self.ds['mask_mre'].loc[{'mask_type': 'combo'}] = combo_mre

        # return ds
        if self.write_file:
            self.output_name = Path(self.data_dir.parents[1], 'XR', f'xarray_{self.output_name}.nc')
            print(f'Writing xr file {self.output_name} ')
            self.ds.to_netcdf(self.output_name)
            print('Done')
            return self.ds

    def resize_image(self, input_image, var_name):
        '''Take an input image and resize it to the appropriate resolution.'''
        # Get initial and resizing params
        init_size = input_image.GetSize()
        init_spacing = input_image.GetSpacing()
        nx = self.nx
        ny = self.ny
        x_change = init_size[0]/self.nx
        y_change = init_size[1]/self.ny

        # Get variable-dependent params
        if 'mri' in var_name:
            nz = self.nz_mri
            z_change = init_size[2]/nz
        else:
            nz = self.nz_mre
            z_change = init_size[2]/nz
        if 'mask' in var_name:
            interp_method = sitk.sitkNearestNeighbor
        else:
            interp_method = sitk.sitkLinear

        # Resize image
        ref_image = sitk.GetImageFromArray(np.ones((nz, ny, nx), dtype=np.uint16))
        ref_image.SetSpacing((init_spacing[0]*x_change, init_spacing[1]*y_change,
                              init_spacing[2]*z_change))
        ref_image.SetOrigin(input_image.GetOrigin())
        ref_image = sitk.Cast(ref_image, input_image.GetPixelIDValue())

        center = sitk.CenteredTransformInitializer(
            ref_image, input_image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        return sitk.Resample(input_image, ref_image, center, interp_method)

    def respace_image(self, input_image, var_name, x_spacing, y_spacing):
        '''Function for changing the spacing of an image, given supplied ideal spacing.
        This is meant for use with the MRE images as they cannot be registered to the inputs.
        '''
        # Get initial params
        nx = self.nx
        ny = self.ny
        z_spacing = input_image.GetSpacing()[-1]

        # Get variable-dependent params
        if 'mri' in var_name:
            nz = self.nz_mri
        else:
            nz = self.nz_mre
        if 'mask' in var_name:
            interp_method = sitk.sitkNearestNeighbor
        else:
            interp_method = sitk.sitkLinear

        # Resize image
        ref_image = sitk.GetImageFromArray(np.ones((nz, ny, nx), dtype=np.uint16))
        ref_image.SetSpacing((x_spacing, y_spacing, z_spacing))
        ref_image.SetOrigin(input_image.GetOrigin())
        ref_image = sitk.Cast(ref_image, input_image.GetPixelIDValue())

        center = sitk.CenteredTransformInitializer(
            ref_image, input_image, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        return sitk.Resample(input_image, ref_image, center, interp_method)

    def gen_liver_mask(self, input_image_np):
        '''Generate the mask of the liver by using the CHAOS segmentation model.'''

        # get the input, force it into the correct shape
        input_image_np = np.where(input_image_np >= 1500, 1500, input_image_np)
        # input_image_np = np.where(input_image_np <= 1e-9, np.nan, input_image_np)
        # mean = np.nanmean(input_image_np)
        # std = np.nanstd(input_image_np)
        # # input_image_np = ((input_image_np - mean)/std) + 4
        # input_image_np = ((input_image_np - mean)/std)
        # input_image_np = np.where(input_image_np != input_image_np, 0, input_image_np)
        # image = np.where(input_image <= 1e-9, np.nan, input_image)

        mean = np.nanmean(input_image_np)
        std = np.nanstd(input_image_np)
        input_image_np = ((input_image_np - mean)/std)
        input_image_np = np.where(input_image_np != input_image_np, 0, input_image_np)

        input_image_np = input_image_np[np.newaxis, np.newaxis, :]
        # get the model prediction (liver mask)
        model_pred = self.model(torch.Tensor(input_image_np))
        model_pred = torch.sigmoid(model_pred)
        # Convert to binary mask
        ones = torch.ones_like(model_pred)
        zeros = torch.zeros_like(model_pred)
        model_pred = torch.where(model_pred > 1e-3, ones, zeros)
        return np.transpose(model_pred.cpu().numpy()[0, 0, :], (2, 1, 0))

    def gen_elast_mask(self, subj):
        '''Generate a mask from the elastMsk, and place it into the given "mre_mask" slot.
        Assumes you are using an xarray dataset from the MREDataset class.'''

        for z in list(self.ds.z_mre):
            # make initial mask from elast and elastMsk
            mre = self.ds['image_mre'].sel(mre_type='mre', z_mre=z, subject=subj).values
            conf = self.ds['image_mre'].sel(mre_type='mre_mask', z_mre=z, subject=subj).values
            msk = np.zeros_like(mre)
            msk = np.where(np.isclose(mre, conf, atol=0.1, rtol=1), 0, 1)
            msk = morphology.binary_dilation(msk)  # fill in little holes

            # invert the mask so that 1s are in and 0s are out
            msk = msk+np.where(mre < 1e-8, 1, 0)
            msk = 1-np.where(msk > 1, 1, msk)

            # place mask into 'mask_mre' slot
            self.ds['mask_mre'].loc[dict(mask_type='mre', z_mre=z, subject=subj)] = msk

    def align_mre_raw(self, fixed, moving, pat):
        pad = np.full((256, 256), 0, np.int16)
        ones = np.full((256, 256), 1, np.int16)
        pad = sitk.GetImageFromArray(pad)
        ones = sitk.GetImageFromArray(ones)

        moving1 = moving[:, :, 0]
        orig = moving1.GetOrigin()
        spacing = moving1.GetSpacing()
        moving2 = moving[:, :, 1]
        moving2.SetOrigin(orig)
        moving3 = moving[:, :, 2]
        moving3.SetOrigin(orig)
        moving4 = moving[:, :, 3]
        moving4.SetOrigin(orig)
        pad.SetOrigin(orig)
        pad.SetSpacing(spacing)
        ones.SetOrigin(orig)
        ones.SetSpacing(spacing)

        with open(Path(self.data_dir, pat, 'mre.pkl'), 'rb') as f:
            mre_location = pkl.load(f)

        pad_nums = np.asarray(np.diff(mre_location)/fixed.GetSpacing()[2], dtype=int)
        # pad_nums = [0, 0, 0]
        # pad_nums = np.ceil(pad_nums)
        # pad_nums = np.asarray(pad_nums, dtype=int)
        pad_start = int(fixed.GetSize()[2]*0.60)
        pad_end = int(fixed.GetSize()[2]*0.40)
        # pad_start = 0
        # pad_end = 0
        moving_new = sitk.JoinSeries([pad]*pad_start + [moving1] + [pad]*pad_nums[0] + [moving2] +
                                     [pad]*pad_nums[1] + [moving3] + [pad]*pad_nums[2] + [moving4] +
                                     [pad]*pad_end)
        moving_new.SetSpacing([moving.GetSpacing()[0], moving.GetSpacing()[1],
                               fixed.GetSpacing()[2]])
        #                       # np.diff(mre_location)[0]])
        moving_mask = sitk.JoinSeries([pad]*pad_start + [ones] + [pad]*pad_nums[0] + [ones] +
                                      [pad]*pad_nums[1] + [ones] + [pad]*pad_nums[2] + [ones] +
                                      [pad]*pad_end)
        moving_mask.SetSpacing([moving.GetSpacing()[0], moving.GetSpacing()[1],
                               fixed.GetSpacing()[2]])
        moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)

        fixed_nums = np.asarray([fixed.GetSize()[2]*0.3, fixed.GetSize()[2]*0.6,
                                 fixed.GetSize()[2]*0.1], dtype=int)
        fixed_mask = sitk.JoinSeries([pad]*fixed_nums[0] + [ones]*fixed_nums[1] +
                                     [pad]*fixed_nums[2])
        fixed_mask.SetSpacing([fixed.GetSpacing()[0], fixed.GetSpacing()[1],
                               fixed.GetSpacing()[2]])
        fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)

        fixed.SetOrigin([fixed.GetOrigin()[0], fixed.GetOrigin()[1],
                         -fixed.GetSize()[2]/2])
        fixed_mask.SetOrigin([fixed.GetOrigin()[0], fixed.GetOrigin()[1],
                              -fixed.GetSize()[2]/2])
        moving_new.SetOrigin([moving_new.GetOrigin()[0], moving_new.GetOrigin()[1],
                              -moving_new.GetSize()[2]])
        moving_mask.SetOrigin([moving_new.GetOrigin()[0], moving_new.GetOrigin()[1],
                              -moving_new.GetSize()[2]])
        moving_mask = None
        # fixed_mask = None
        reg = Register(fixed, moving_new, dry_run=False, config='mre_match',
                       fixed_mask=fixed_mask, moving_mask=moving_mask)

        np_res = sitk.GetArrayFromImage(reg.moving_img_result)
        std_dev = np_res.std(axis=(1, 2))
        peaks, _ = find_peaks(std_dev, height=(np.mean(std_dev), None))
        if len(peaks) != 4:
            peaks = sorted(std_dev.argsort()[-4:])
        # plt.bar(range(np_res.shape[0]), std_dev, label='Std Dev')
        # plt.plot(peaks, (std_dev)[peaks], 'x', c='C1', markersize=10, mew=5,
        # label='Peak Location')
        # plt.title('Std Dev of each moving img slice')
        # plt.xlabel('Slice Location')
        # plt.ylabel('Std Dev')
        # plt.legend()
        # plt.show()
        return peaks

    def reg_inputs(self, reg_pat):
        for seq in self.sequences:
            if seq == self.primary_input:
                continue
            elif seq not in reg_pat.images.keys():
                continue

            np_tmp = sitk.GetArrayFromImage(reg_pat.images[seq])
            mov_min = float(np_tmp.min())
            mov_max = float(np_tmp.max())
            reg = Register(reg_pat.images[self.primary_input], reg_pat.images[seq],
                           config='mri_seq')
            reg.moving_img_result = sitk.RescaleIntensity(
                reg.moving_img_result, mov_min, mov_max)

            resized_image = self.resize_image(reg.moving_img_result, 'input_mri')

            self.ds['image_mri'].loc[{'sequence': seq}] = (sitk.GetArrayFromImage(resized_image).T)

            # Do the mre_raw alignment
            if seq == 't1_pre_in':
                self.mri_to_mre_idx = self.align_mre_raw(resized_image, reg_pat.images['mre_raw'],
                                                         reg_pat.subj)
                resized_t1_pre = resized_image

        resized_primary = self.resize_image(reg_pat.images[self.primary_input], 'input_mri')
        self.ds['image_mri'].loc[{'sequence': self.primary_input}] = (
            sitk.GetArrayFromImage(resized_primary).T)

        return resized_t1_pre, resized_primary

    def _check_ipython(self):
        # from: https://stackoverflow.com/questions/15341757/
        # how-to-check-that-pylab-backend-of-matplotlib-runs-inline/17826459#17826459
        try:
            cfg = get_ipython().config  # noqa: F821
            print('Called by IPython.')

            # Caution: cfg is an IPython.config.loader.Config
            if cfg['IPKernelApp']:
                print('Within IPython QtConsole.')

                try:
                    if cfg['IPKernelApp']['pylab'] == 'inline':
                        print('inline pylab loaded.')
                    else:
                        print('pylab loaded, but not in inline mode.')
                except NameError:
                    print('pylab not loaded.')
            elif cfg['TerminalIPythonApp']:
                try:
                    if cfg['TerminalIPythonApp']['pylab'] == 'inline':
                        print('inline pylab loaded.')
                    else:
                        print('pylab loaded, but not in inline mode.')
                except NameError:
                    print('pylab not loaded.')
            return True
        except NameError:
            print('Not called by IPython.')
            return False
