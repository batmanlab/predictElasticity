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
import PIL
import pdb
from tqdm import tqdm_notebook
# import matplotlib.pyplot as plt
# import holoviews as hv

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter, median_filter

from mre.registration import RegPatient, Register
from mre.pytorch_arch_old import GeneralUNet3D
from mre.pytorch_arch_deeplab import DeepLab
from mre.ord_bins import get_ord_binning


class MREtoXr:
    '''Make an xr dataset for mre nifti data.  If some patients are missing data, that data is added
    as a 0'd vector.  Includes two coordinate systems (one for MRI data, one for MRE data).
    Includes indicators for data quality. Processes single patient at a time.
    '''
    def load_files(self, file_names):
        if type(file_names) is not list:
            file_names = str(file_names)
        else:
            file_names = [f for f in file_names if Path(f).exists()]

        if '*' in file_names or type(file_names) is list:
            if re.match(r'_n\*_', file_names):
                ds = self.get_best_data(file_names)
            else:
                ds = xr.open_mfdataset(file_names, combine='nested', concat_dim='subject')
        else:
            ds = xr.open_dataset(file_names)
        return ds

    def get_best_data(self, file_names):
        pass
        # versions = []
        # groups = []
        # slurm_config_dir = '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/config/'
        # config_path = Path(slurm_config_dir)
        # for i in Path(test_path).parents[2].glob(Path(test_path).parents[1].stem):
        #     versions.append(str(i).split('_')[-1])

    def __init__(self, data_dir=None, sequences=None, patient=None, from_file=None, **kwargs):

        self.is_ipython = self._check_ipython()
        if None is data_dir is sequences is from_file:
            raise ValueError(
                '(data_dir and sequences) or (from_file) must be specified to initialize')

        if from_file:
            from_file_pred = kwargs.get('from_file_pred', None)
            if not from_file_pred:
                self.ds = self.load_files(from_file)
                # self.ds = self.ds.load()
                return None
            else:
                ds_pred = self.load_files(from_file_pred)

                # Detect if using old-style predictions (not separated from input)
                if 'image_mri' in ds_pred:
                    self.ds = ds_pred
                    self.ds = self.ds.load()
                # Otherwise stitch pred together with input images
                else:
                    self.ds = self.load_files(from_file)
                    print(ds_pred.subject.values)
                    print(self.ds.subject.values)
                    self.ds = self.ds.sel(subject=ds_pred.subject.values)
                    self.ds = self.ds.load()
                    self.ds['image_mre'].loc[
                        dict(mre_type='mre_pred', subject=ds_pred.subject)] = ds_pred['image_mre']

                    if 'val_slope' in ds_pred and 'val_intercept' in ds_pred:
                        self.ds['val_slope'] = (('subject', np.zeros(len(self.ds.subject))))
                        self.ds['val_slope'].loc[
                            dict(subject=ds_pred.subject)] = ds_pred['val_slope']
                        self.ds['val_intercept'] = (('subject', np.zeros(len(self.ds.subject))))
                        self.ds['val_intercept'].loc[
                            dict(subject=ds_pred.subject)] = ds_pred['val_intercept']

            return None

        elif data_dir:
            self.data_dir = data_dir
            if sequences is None:
                sequences = ['t1_pre_water', 't1_pre_in', 't1_pre_out', 't1_pre_fat', 't2',
                             't1_pos_0_water', 't1_pos_70_water', 't1_pos_160_water',
                             't1_pos_300_water']
            self.sequences = sequences
            if len(self.sequences) == 0:
                raise ValueError('No sequences specificed')
            if patient is None:
                self.patient = '0006'
            else:
                self.patient = str(patient)

            # Check if any contrast images are specified
            if np.any(['pos' in seq for seq in self.sequences]):
                # out_subdir = 'XR_with_contrast_v2'
                # out_subdir = 'XR_resized'
                out_subdir = 'XR_v3'
            else:
                out_subdir = 'XR'
            self.output_dir = Path(self.data_dir.parents[1], out_subdir)
            print(self.data_dir)
            print(self.patient)
            print(self.sequences)
            print(self.output_dir)
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

        # OLD VERSION
        # model_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/',
        #                   'trained_models', '01', 'model_2019-11-06_18-27-49.pkl')
        # with torch.no_grad():
        #     self.model = GeneralUNet3D(5, 1, 24, 1, True, False, False)
        #     model_dict = torch.load(model_path, map_location='cpu')
        #     model_dict = OrderedDict([(key[7:], val) for key, val in model_dict.items()])
        #     self.model.load_state_dict(model_dict, strict=True)
        #     self.model.eval()

        # NEW VERSION
        model_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/',
                          'trained_models', '001', 'model_2020-02-12_14-14-16.pkl')
        with torch.no_grad():
            self.model = DeepLab(1, 1, output_stride=8, do_ord=False)
            model_dict = torch.load(model_path, map_location='cpu')
            model_dict = OrderedDict([(key[7:], val) for key, val in model_dict.items()])
            self.model.load_state_dict(model_dict, strict=True)
            self.model.eval()

        # Initialize empty ds
        self.init_new_ds()

    def get_ds(self):
        '''Return the ds loaded via 'from_file'.'''
        return self.ds

    def get_trimmed_ds(self):
        self.ds = self.ds.load()
        ds_subj_list = []
        for subj in tqdm_notebook(self.ds.subject.values):
            ds_subj = self.ds.sel(subject=[subj])
            self.ds = self.ds.drop(subj, dim='subject')
            ds_subj = ds_subj.sel(z_mri=ds_subj.mri_to_mre_idx.values.flatten())
            ds_subj.assign_coords(z_mri=[0, 1, 2, 3])
            ds_subj_list.append(ds_subj)
        self.ds = xr.merge(ds_subj_list)
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
        liver_input = self.ds['image_mri'].loc[{'subject': self.patient, 'sequence': 't1_pre_in'}]
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
        combo_mri = morphology.remove_small_objects(combo_mri.astype(bool), 400)
        self.ds['mask_mri'].loc[{'mask_type': 'combo'}] = combo_mri.astype(int)

        combo_mre = (self.ds['mask_mre'].sel(subject=self.patient, mask_type='mre').values *
                     self.ds['mask_mre'].sel(subject=self.patient, mask_type='liver').values)
        # combo_mre = morphology.binary_erosion(combo_mre.values)
        # combo_mre = morphology.binary_erosion(combo_mre)
        combo_mre = combo_mre.astype(bool)
        for z in range(combo_mre.shape[-1]):
            combo_mre[:, :, z] = morphology.remove_small_objects(combo_mre[:, :, z], 400)
        self.ds['mask_mre'].loc[{'mask_type': 'combo'}] = combo_mre.astype(int)

        # return ds
        if self.write_file:
            self.output_name = Path(self.output_dir, f'xarray_{self.output_name}.nc')
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
        # x_change = init_size[0]/self.nx
        # y_change = init_size[1]/self.ny

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
        # ref_image.SetSpacing((init_spacing[0]*x_change, init_spacing[1]*y_change,
        #                       init_spacing[2]*z_change))
        if 'mri' in var_name:
            # ref_image.SetSpacing((1.875, 1.875, 10.063))
            ref_image.SetSpacing((1.875, 1.875, init_spacing[2]*z_change))
        else:
            ref_image.SetSpacing((1.875, 1.875, init_spacing[2]*z_change))
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
        with torch.no_grad():
            print('starting torch model loading')
            model_pred = self.model(torch.Tensor(input_image_np))
            print('sigmoid func')
            model_pred = torch.sigmoid(model_pred)
            print('torch done')
            # Convert to binary mask
            # print('ones and zeros')
            # ones = torch.ones_like(model_pred)
            # zeros = torch.zeros_like(model_pred)
            # print('where')
            # model_pred = torch.where(model_pred > 1e-1, ones, zeros)
        output_image_np = np.transpose(model_pred.cpu().numpy()[0, 0, :], (2, 1, 0))
        output_image_np = np.where(output_image_np > 0.5, 1, 0)
        return output_image_np

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


class MRETorchDataset(Dataset):
    '''Make a torch dataset compatible with the torch dataloader.'''
    def __init__(self, xa_ds, set_type, **kwargs):

        # Required params
        self.xa_ds = xa_ds
        self.set_type = set_type

        # Assign kwargs
        self.dims = kwargs.get('dims', 2)
        self.seed = kwargs.get('seed', 100)
        self.inputs = kwargs.get('inputs', ['t1_pre_water', 't1_pre_in', 't1_pre_out', 't1_pre_fat',
                                            't2', 't1_pos_0_water', 't1_pos_70_water',
                                            't1_pos_160_water', 't1_pos_300_water'])
        if type(self.inputs) is str:
            self.inputs = [self.inputs]
        self.target = kwargs.get('target', 'mre')
        self.mask = kwargs.get('mask', 'combo')
        self.clip = kwargs.get(f'{set_type}_clip', True)
        self.transform = kwargs.get(f'{set_type}_transform', True)
        self.aug = kwargs.get(f'{set_type}_aug', False)
        self.smear = kwargs.get(f'{set_type}_smear', False)
        self.smear_amt = kwargs.get('smear_amt', 0)
        self.loss = kwargs.get(f'loss', 'l2')
        self.bins = kwargs.get(f'bins', None)
        self.nbins = kwargs.get(f'out_channels_final', 0)
        self.organize_data()

    def organize_data(self):
        '''Reorder, seperate and manipulate input data such that it conforms to dataloader
        standards and required format given a particular training task.'''

        np.random.seed(self.seed)
        if self.set_type != 'eval':
            print(self.xa_ds.subject)

        if self.dims == 2:
            # stack subject and z-slices to make 4 2D image groups for each 3D image group
            self.xa_ds = self.xa_ds.stack(subject_2d=('subject', 'z')).reset_index('subject_2d')
            subj_2d_coords = [f'{i.subject.values}_{i.z.values}' for i in self.xa_ds.subject_2d]
            self.xa_ds = self.xa_ds.assign_coords(subject_2d=subj_2d_coords)

            self.name_dict = dict(zip(range(len(self.xa_ds.subject_2d)),
                                      self.xa_ds.subject_2d.values))

            self.input_images = self.xa_ds.sel(sequence=self.inputs).image_mri.transpose(
                'subject_2d', 'sequence', 'y', 'x').values
            self.target_images = self.xa_ds.sel(mre_type=[self.target]).image_mre.transpose(
                'subject_2d', 'mre_type', 'y', 'x').values
            self.mask_images = self.xa_ds.sel(mask_type=[self.mask]).mask_mre.transpose(
                'subject_2d', 'mask_type', 'y', 'x').values

            self.names = self.xa_ds.subject_2d.values

        elif self.dims == 3:
            print(self.inputs)
            self.input_images = self.xa_ds.sel(sequence=self.inputs).image_mri.transpose(
                'subject', 'sequence', 'z', 'y', 'x')
            self.target_images = self.xa_ds.sel(mre_type=[self.target]).image_mre.transpose(
                'subject', 'mre_type', 'z', 'y', 'x')
            self.mask_images = self.xa_ds.sel(mask_type=[self.mask]).mask_mre.transpose(
                'subject', 'mask_type', 'z', 'y', 'x')

            self.names = self.xa_ds.subject.values

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        mask = self.mask_images[idx].values.astype(np.float32)
        image = self.input_images[idx].values
        target = self.target_images[idx].values
        if self.clip:
            image  = np.where(image >= 2000, 2000, image)
            if self.loss == 'l2':
                with np.errstate(divide='ignore', invalid='ignore'):
                    # target = np.float32(np.where(target > 0, np.sqrt(target), 0))
                    target = np.float32(target/100.0)
            target = np.float32(target)

        if self.dims == 2:
            image, target, mask = self.get_data_aug_2d(image, target, mask)
        elif self.dims == 3:
            image, target, mask = self.get_data_aug_3d(image, target, mask)

        return [image, target, mask, self.names[idx]]

    def get_data_aug_2d(self, image, target, mask):
        if self.transform:
            if self.aug:
                rot_angle = np.random.uniform(-8, 8, 1)
                translations = np.random.uniform(-10, 10, 2)
                scale = np.random.uniform(0.90, 1.10, 1)
                if self.smear == 'gaussian':
                    # sigma = np.random.uniform(0, 3, 1)[0]
                    sigma = self.smear_amt
                elif self.smear == 'median':
                    # sigma = np.random.randint(10)
                    # sigma = 6
                    sigma = self.smear_amt
                else:
                    sigma = 0
            else:
                rot_angle = 0
                translations = (0, 0)
                scale = 1
                sigma = 0

            image = self.input_norm(image)
            img_list = []
            # Iterate over channels
            for i in range(image.shape[0]):
                img_list.append(self.affine_transform(image[i], rot_angle, translations, scale,
                                                      resample=PIL.Image.BILINEAR))
            image = torch.cat(img_list)

            mask = self.affine_transform(mask[0], rot_angle, translations, scale,
                                         resample=PIL.Image.NEAREST)
            if self.smear == 'gaussian':
                target = gaussian_filter(target[0], sigma=sigma)
            elif self.smear == 'median':
                target = median_filter(target[0], sigma=sigma)
            target = self.affine_transform(target, rot_angle, translations, scale,
                                           resample=PIL.Image.BILINEAR)

        image = torch.FloatTensor(image)
        target = torch.FloatTensor(target)
        mask = torch.FloatTensor(mask)

        return image, target, mask

    def get_data_aug_3d(self, image, target, mask):
        if self.transform:
            if self.aug:
                rot_angle_xy = np.random.uniform(-8, 8, 1)[0]
                translations_xy = np.random.uniform(-10, 10, 2)

                # rot_angle_xz = np.random.uniform(-1, 1, 1)[0]
                # rot_angle_yz = np.random.uniform(-1, 1, 1)[0]

                scale = np.random.uniform(0.90, 1.10, 1)[0]

                if self.smear == 'gaussian':
                    # sigma = np.random.uniform(0, 3, 1)[0]
                    sigma = self.smear_amt
                elif self.smear == 'median':
                    # sigma = np.random.randint(1, 10)
                    # sigma = np.random.randint(1, 6)
                    # sigma = 4
                    if self.smear_amt == -1:
                        sigma = np.random.randint(1, 6)
                    else:
                        sigma = self.smear_amt
                else:
                    sigma = 0
            else:
                # raise NotImplementedError('you must transform 3d images')
                rot_angle_xy = 0
                translations_xy = (0, 0)

                # rot_angle_xz = 0
                # rot_angle_yz = 0

                scale = 1
                sigma = 0

            image = self.input_norm(image)
            img_list = []
            # Iterate over channels
            for i in range(image.shape[0]):
                slice_list = []
                # Iterate over z-slice
                for j in range(image.shape[1]):
                    slice_list.append(self.affine_transform(image[i][j], rot_angle_xy,
                                                            translations_xy, scale,
                                                            resample=PIL.Image.BILINEAR))
                img_list.append(torch.cat(slice_list))
            image = torch.stack(img_list)

            # Iterate over z-slice
            mask_list = []
            target_list = []
            for j in range(mask.shape[1]):
                mask_list.append(self.affine_transform(mask[0][j], rot_angle_xy, translations_xy,
                                                       scale, resample=PIL.Image.NEAREST))
                if self.smear == 'guassian':
                    target_tmp = gaussian_filter(target[0][j], sigma=sigma)
                elif self.smear == 'median':
                    target_tmp = median_filter(target[0][j], size=sigma)
                else:
                    target_tmp = target[0][j]
                target_list.append(self.affine_transform(target_tmp, rot_angle_xy,
                                                         translations_xy, scale,
                                                         resample=PIL.Image.BILINEAR))
            mask = torch.cat(mask_list)
            target = torch.cat(target_list)

        image = torch.FloatTensor(image)
        # target = torch.FloatTensor(target)
        if self.loss == 'ordinal':
            bins, _, _ = get_ord_binning(self.bins, self.nbins)
            bins[-1] = 1e6
            bins[0] = 0
            target = np.digitize(target, bins)
        # target = torch.IntTensor(target.type(torch.IntTensor))
        target = torch.FloatTensor(target)
        target = target.unsqueeze(0)
        mask = torch.FloatTensor(mask)
        mask = mask.unsqueeze(0)

        return image, target, mask

    def affine_transform(self, input_slice, rot_angle=0, translations=0, scale=1, resample=None):
        input_slice = transforms.ToPILImage()(input_slice)
        input_slice = TF.affine(input_slice, angle=rot_angle,
                                translate=list(translations), scale=scale, shear=0,
                                resample=resample)
        input_slice = transforms.ToTensor()(input_slice)
        return input_slice

    def affine_transform_3d(self, image, rot_angle_xy=None, rot_angle_xz=None, rot_angle_yz=None,
                            translations_xy=None, scale=None, order=None):
        image = ndi.interpolation.rotate(image, rot_angle_xy, axes=(1, 2), order=order,
                                         reshape=False)
        # image = ndi.interpolation.rotate(image, rot_angle_xz, axes=(0, 2), order=order,
        #                                  reshape=False)
        # image = ndi.interpolation.rotate(image, rot_angle_yz, axes=(0, 1), order=order,
        #                                  reshape=False)
        image = ndi.interpolation.shift(image, shift=[0, translations_xy[1], translations_xy[0]],
                                        order=order)
        # image = ndi.interpolation.zoom(image, zoom=scale, order=order)
        return image

    def input_norm(self, input_image, rot_angle=0, translations=0, scale=1, resample=None):

        # normalize image
        if self.dims == 2:
            image = input_image
            image = np.where(input_image <= 1e-9, np.nan, input_image)
            mean = np.nanmean(image, axis=(1, 2))
            std = np.nanstd(image, axis=(1, 2))
            image = ((image.T - mean)/std).T
            image = np.where(image != image, 0, image)
            image = image.astype(np.float32)
        elif self.dims == 3:
            image = input_image
            image = np.where(input_image <= 1e-9, np.nan, input_image)
            mean = np.nanmean(image, axis=(1, 2, 3))
            std = np.nanstd(image, axis=(1, 2, 3))
            image = ((image.T - mean)/std).T
            image = np.where(image != image, 0, image)
            image = image.astype(np.float32)
        return image


class TorchToXr:
    '''Make an xr dataset for mre torch inputs directly from the output of a dataloader.  This is
    specifically for feeding into the xr_viewer_v2 for debugging purposes.  This views the input
    images, masks, and targets right before feeding them into an ML model.
    '''
    def __init__(self, images, masks, targets, names, **kwargs):
        images = images.numpy()
        masks = masks.numpy()
        targets = targets.numpy()
        if len(images.shape) == 5:
            self.dims = 3
        elif len(images.shape) == 4:
            self.dims = 2
        else:
            raise ValueError(f"images are wrong shape: {images.shape}")

        # Required params
        if images.shape[0] != masks.shape[0] != targets.shape[0] != len(names):
            print(f'images: {images.shape}')
            print(f'masks: {masks.shape}')
            print(f'targets: {targets.shape}')
            print(f'names: {len(names)}')
            raise ValueError('Number of subjects not equal.')
        if (images.shape[2:] != masks.shape[2:] != targets.shape[2:]):
            print(f'images: {images.shape}')
            print(f'masks: {masks.shape}')
            print(f'targets: {targets.shape}')
            raise ValueError('Image shapes not equal.')
        if self.dims == 3:
            self.nx = images.shape[4]
            self.ny = images.shape[3]
            self.nz = images.shape[2]
        elif self.dims == 2:
            self.nx = images.shape[3]
            self.ny = images.shape[2]
        self.nseq = images.shape[1]
        self.nsub = len(names)
        self.names = list(names)

        # Assign kwargs
        self.sequences = kwargs.get('sequences', ['t1_pre_water', 't1_pre_in', 't1_pre_out',
                                                  't1_pre_fat', 't2', 't1_pos_0_water',
                                                  't1_pos_70_water', 't1_pos_160_water',
                                                  't1_pos_300_water'])
        self.init_new_ds()
        if self.dims == 3:
            self.ds['image_mri'] = (('subject', 'sequence', 'z', 'y', 'x'), images)
            self.ds['image_mre'] = (('subject', 'mre_type', 'z', 'y', 'x'), targets)
            self.ds['mask_mri'] = (('subject', 'mask_type', 'z', 'y', 'x'), masks)
            self.ds['mask_mre'] = (('subject', 'mask_type', 'z', 'y', 'x'), masks)
        elif self.dims == 2:
            self.ds['image_mri'] = (('subject', 'sequence', 'y', 'x'), images)
            self.ds['image_mre'] = (('subject', 'mre_type', 'y', 'x'), targets)
            self.ds['mask_mri'] = (('subject', 'mask_type', 'y', 'x'), masks)
            self.ds['mask_mre'] = (('subject', 'mask_type', 'y', 'x'), masks)

    def init_new_ds(self):
        '''Initialize a new xarray dataset based on the size and shape of our inputs.'''

        if self.dims == 3:
            self.ds = xr.Dataset(
                {'image_mri': (['subject', 'sequence', 'z', 'y', 'x'],
                               np.zeros((self.nsub, self.nseq, self.nz, self.ny,
                                         self.nx), dtype=np.float32)),
                 'image_mre': (['subject', 'mre_type', 'z', 'y', 'x'],
                               np.zeros((self.nsub, 1, self.nz, self.ny,
                                         self.nx), dtype=np.float32)),
                 'mask_mre': (['subject', 'mask_type', 'z', 'y', 'x'],
                              np.zeros((self.nsub, 1, self.nz, self.ny,
                                        self.nx), dtype=np.float32)),
                 'mask_mri': (['subject', 'mask_type', 'z', 'y', 'x'],
                              np.zeros((self.nsub, 1, self.nz, self.ny,
                                        self.nx), dtype=np.float32)),
                 },

                coords={'subject': self.names,
                        'sequence': self.sequences,
                        'mask_type': ['combo'],
                        'mre_type': ['mre'],
                        'x': range(self.nx),
                        'y': range(self.ny)[::-1],
                        'z': range(self.nz),
                        }
            )
        else:

            self.ds = xr.Dataset(
                {'image_mri': (['subject', 'sequence', 'y', 'x'],
                               np.zeros((self.nsub, self.nseq, self.ny,
                                         self.nx), dtype=np.float32)),
                 'image_mre': (['subject', 'mre_type', 'y', 'x'],
                               np.zeros((self.nsub, 1, self.ny,
                                         self.nx), dtype=np.float32)),
                 'mask_mre': (['subject', 'mask_type', 'y', 'x'],
                              np.zeros((self.nsub, 1, self.ny,
                                        self.nx), dtype=np.float32)),
                 'mask_mri': (['subject', 'mask_type', 'y', 'x'],
                              np.zeros((self.nsub, 1, self.ny,
                                        self.nx), dtype=np.float32)),
                 },

                coords={'subject': self.names,
                        'sequence': self.sequences,
                        'mask_type': ['combo'],
                        'mre_type': ['mre'],
                        'x': range(self.nx),
                        'y': range(self.ny)[::-1],
                        }
            )


class ModelCompare:
    '''
    Simple Class for making an xarray made for many model predictions.
    '''

    def __init__(self, base_model_path='/pghbio/dbmi/batmanlab/Data/MRE/XR_full_gold_v3/*.nc',
                 pred_dict=None):

        if pred_dict is None:
            pred_dict = {}

        self.init_new_ds(base_model_path, pred_dict.keys())
        self.add_predictions(pred_dict)

    def init_new_ds(self, base_model_path, pred_names):
        '''Initialize a new xarray dataset based on the size and shape of our inputs.'''

        # Generate an xarray dataset from a saved base model and append all the saved predictions
        # All 4 data vars are 5D (subject, sequence/mask_type, x, y, z).

        print(base_model_path)
        base_ds = self.load_files(base_model_path)

        mre_type = list(base_ds.mre_type.values)
        mre_type.remove('mre_pred')
        mre_type += pred_names
        # mre_type
        self.ds = xr.Dataset(
            {'image_mri': (['subject', 'sequence', 'x', 'y', 'z'],
                           base_ds['image_mri']),
             'mask_mri': (['subject', 'mask_type', 'x', 'y', 'z'],
                          base_ds['mask_mri']),
             'image_mre': (['subject', 'mre_type', 'x', 'y', 'z'],
                           np.zeros((len(base_ds.subject),
                                     len(mre_type),
                                     len(base_ds.x), len(base_ds.y), len(base_ds.z)),
                                    dtype=np.float)),
             'mask_mre': (['subject', 'mask_type', 'x', 'y', 'z'],
                          base_ds['mask_mre']),
             'mri_to_mre_idx': (['subject', 'z_idx'],
                                base_ds['mri_to_mre_idx']),
             'val_slope': (['subject', 'mre_type'],
                           np.ones((len(base_ds.subject),
                                    len(mre_type)),
                                   dtype=np.float)),
             'val_intercept': (['subject', 'mre_type'],
                               np.zeros((len(base_ds.subject),
                                         len(mre_type)),
                                        dtype=np.float)),
             },

            coords={'subject': base_ds.subject,
                    'sequence': base_ds.sequence,
                    'mask_type': base_ds.mask_type,
                    'mre_type': mre_type,
                    'x': base_ds.x,
                    'y': base_ds.y,
                    'z': base_ds.z
                    }
        )
        # Fill in the model mre_types
        self.ds['image_mre'].loc[
            {'mre_type': 'mre'}] = base_ds['image_mre'].sel(mre_type='mre')
        self.ds['image_mre'].loc[
            {'mre_type': 'mre_wave'}] = base_ds['image_mre'].sel(mre_type='mre_wave')
        self.ds['image_mre'].loc[
            {'mre_type': 'mre_mask'}] = base_ds['image_mre'].sel(mre_type='mre_mask')
        self.ds['image_mre'].loc[
            {'mre_type': 'mre_raw'}] = base_ds['image_mre'].sel(mre_type='mre_raw')

    def load_files(self, file_names):
        if type(file_names) is not list:
            file_names = str(file_names)
        else:
            file_names = [f for f in file_names if Path(f).exists()]

        if '*' in file_names or type(file_names) is list:
            ds = xr.open_mfdataset(file_names, combine='nested', concat_dim='subject')
        else:
            ds = xr.open_dataset(file_names)
        return ds

    def add_predictions(self, pred_dict):
        '''Append predictions to self.ds'''
        for pred in pred_dict:
            ds_pred = self.load_files(pred_dict[pred])
            ds_pred = ds_pred.assign_coords(mre_type=[pred])
            ds_pred = ds_pred.sortby('subject')
            print(ds_pred)
            self.ds['image_mre'].loc[dict(mre_type=pred)] = ds_pred['image_mre']
            self.ds['val_slope'].loc[dict(mre_type=pred)] = ds_pred['val_slope']
            self.ds['val_intercept'].loc[dict(mre_type=pred)] = ds_pred['val_intercept']


class ModelComparePandas:
    '''
    Simple Class for converting the model compare xarray to a pandas Dataframe.
    Assumes input is the ModelCompare Xarray
    '''

    def __init__(self, ds):
        pred_names = [pred for pred in list(ds.mre_type.values) if
                      pred not in ['mre_raw', 'mre_mask', 'mre_pred', 'mre_wave']]
        pred_dict = {}
        for pred in pred_names:
            pred_dict[pred] = []
            for subj in ds.subject:
                mask = ds.sel(subject=subj, mask_type='combo')['mask_mre'].values
                mask = np.where(mask > 0, mask, np.nan)
                pred_mre_region = (ds.sel(subject=subj, mre_type=pred)['image_mre'].values * mask)
                pred_mre_region = pred_mre_region.flatten()
                pred_mre_region = pred_mre_region[~np.isnan(pred_mre_region)]
                # if do_cor:
                #     slope = np.mean(ds.sel(subject=subj)['val_slope'].values)
                #     intercept = np.mean(ds.sel(subject=subj)['val_intercept'].values)
                #     # print(slope, intercept)
                #     pred_mre_region = (pred_mre_region-intercept)/slope
                #     pred_mre_region = np.where(pred_mre_region > 0, pred_mre_region, 0)

                pred_dict[pred].append(np.nan_to_num(np.nanmean(pred_mre_region)))

        self.df = pd.DataFrame(pred_dict, index=ds.subject.values)
