import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
from datetime import datetime
from scipy import ndimage as ndi
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology
import glob


class MRELiverMask:
    '''Class that generates liver masks for MRE input images'''

    def __init__(self, fixed_subj, moving_subj, fixed_seq='T2SS', moving_seq='T1DUAL',
                 fixed_path=None, moving_path=None, verbose=False):

        self.verbose = verbose
        if fixed_path is None:
            self.fixed_path = '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/MRE/'
        else:
            self.fixed_path = fixed_path
        if moving_path is None:
            self.moving_path = ('/pghbio/dbmi/batmanlab/Data/'
                                'CombinedHealthyAbdominalOrganSegmentation/MR_data_batch1/')
        else:
            self.moving_path = moving_path

        self.fixed_seq = fixed_seq
        self.moving_seq = moving_seq
        self.fixed_subj = fixed_subj
        self.moving_subj = moving_subj

        self.load_fixed()
        self.load_moving()
        self.load_moving_mask()

    def load_fixed(self):
        nifti_name = self.fixed_path + '/' + self.fixed_subj + '/' + self.fixed_seq + '.nii'

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(nifti_name)
        img = reader.Execute()
        size = img.GetSize()
        dims = img.GetSpacing()
        orig = img.GetOrigin()
        if self.verbose:
            print(f"Fixed Image info for {nifti_name}:")
            print("Image size:", size[0], size[1], size[2])
            print("Image dims:", dims[0], dims[1], dims[2])
            print("Image orig:", orig[0], orig[1], orig[2])

        fixed_img = img
        fixed_img.GetPixelIDTypeAsString()
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkFloat32)
        self.fixed_img = caster.Execute(fixed_img)

    def load_moving(self):
        dcm_names = '/'.join([self.moving_path, self.moving_subj, self.moving_seq,
                              '/DICOM_anon/*dcm'])
        moving_img_list = []

        if self.verbose:
            print(f"Moving Image info for {dcm_names}")

        for i, fn in enumerate(sorted(glob.glob(dcm_names))):
            if 'DUAL' in self.moving_seq:
                if i % 2 == 0:
                    continue
            if i == 1:
                # grab a single image to copy its metadata
                tmp_moving_img = sitk.ReadImage(fn)
            moving_img_list.append(sitk.GetArrayFromImage(sitk.ReadImage(fn)))

        moving_img_array = np.zeros((len(moving_img_list), moving_img_list[0].shape[1],
                                     moving_img_list[0].shape[2]), dtype=np.float32)
        for i, img in enumerate(moving_img_list):
            moving_img_array[i, :, :] = img

        self.moving_img = sitk.GetImageFromArray(moving_img_array)
        self.moving_img.SetDirection(tmp_moving_img.GetDirection())
        self.moving_img.SetOrigin(tmp_moving_img.GetOrigin())
        self.moving_img.SetSpacing(tmp_moving_img.GetSpacing())
        size = self.moving_img.GetSize()
        dims = self.moving_img.GetSpacing()
        orig = self.moving_img.GetOrigin()
        if self.verbose:
            print("Image size:", size[0], size[1], size[2])
            print("Image dims:", dims[0], dims[1], dims[2])
            print("Image orig:", orig[0], orig[1], orig[2])

    def load_moving_mask(self):
        png_names = '/'.join([self.moving_path, self.moving_subj, self.moving_seq,
                              '/Ground/*png'])
        seg_img_list = []
        for i, fn in enumerate(sorted(glob.glob(png_names))):
            tmp_seg_img = sitk.GetArrayFromImage(sitk.ReadImage(fn))
            # mask out all organs besides the liver (val = 80)
            tmp_seg_img = np.where((tmp_seg_img > 79) & (tmp_seg_img < 81), 80, 0)
            seg_img_list.append(tmp_seg_img)

        seg_img_array = np.zeros((len(seg_img_list), seg_img_list[0].shape[0],
                                  seg_img_list[0].shape[1]), dtype=np.uint8)
        for i, img in enumerate(seg_img_list):
            seg_img_array[i, :, :] = img

        self.moving_mask = sitk.GetImageFromArray(seg_img_array)
        self.moving_mask.SetDirection(self.moving_img.GetDirection())
        self.moving_mask.SetOrigin(self.moving_img.GetOrigin())
        self.moving_mask.SetSpacing(self.moving_img.GetSpacing())

    def gen_param_map(self):
        self.p_map_vector = sitk.VectorOfParameterMap()
        paff = sitk.GetDefaultParameterMap("affine")
        pbsp = sitk.GetDefaultParameterMap("bspline")
        # paff['NumberOfSamplesForExactGradient'] = ['8192']
        # pbsp['NumberOfSamplesForExactGradient'] = ['8192']
        # paff['MaximumNumberOfSamplingAttempts'] = ['2']
        # pbsp['MaximumNumberOfSamplingAttempts'] = ['2']
        # paff['NumberOfSpatialSamples'] = ['4096']
        # pbsp['NumberOfSpatialSamples'] = ['4096']
        # paff['MaximumNumberOfIterations'] = ['512']
        # pbsp['MaximumNumberOfIterations'] = ['512']
        # paff['NumberOfResolutions'] = ['4']
        # pbsp['NumberOfResolutions'] = ['4']
        # paff['GridSpacingSchedule'] = ['2.803220', '1.988100', '1.410000', '1.000000']
        # pbsp['GridSpacingSchedule'] = ['2.803220', '1.988100', '1.410000', '1.000000']
        # pbsp['FinalGridSpacingInPhysicalUnits'] = ['16', '16', '8']
        self.p_map_vector.append(paff)
        self.p_map_vector.append(pbsp)
        if self.verbose:
            sitk.PrintParameterMap(self.p_map_vector)

    def register_imgs(self):
        self.elastixImageFilter = sitk.ElastixImageFilter()
        self.elastixImageFilter.SetFixedImage(self.fixed_img)
        self.elastixImageFilter.SetMovingImage(self.moving_img)
        self.elastixImageFilter.SetParameterMap(self.p_map_vector)
        self.elastixImageFilter.Execute()
        self.moving_img_result = self.elastixImageFilter.GetResultImage()
        self.moving_img_result.CopyInformation(self.fixed_img)

    def gen_mask(self):
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(
            self.elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetMovingImage(self.moving_mask)
        transformixImageFilter.Execute()
        self.moving_mask_result = transformixImageFilter.GetResultImage()
        self.moving_mask_result.CopyInformation(self.fixed_img)
