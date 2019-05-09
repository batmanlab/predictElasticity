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
from tqdm import tqdm


class MRELiverMask:
    '''Class that generates liver masks for MRE input images'''

    def __init__(self, fixed_subj, moving_subj, fixed_seq='T2SS', moving_seq='T1DUAL',
                 fixed_path=None, moving_path=None, verbose=False, center=False, cut=0):

        self.verbose = verbose
        if fixed_path is None:
            self.fixed_path = '/pghbio/dbmi/batmanlab/Data/MRE/'
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

        self.load_fixed(center)
        # self.load_fixed(center, extra=True)
        # self.load_moving(center)
        self.load_moving_nifti(cut=cut)
        # self.load_moving_mask()
        self.load_moving_mask_nifti(cut=cut)

    def load_fixed(self, center, extra=None):
        if extra is None:
            nifti_name = self.fixed_path + '/' + self.fixed_subj + '/' + self.fixed_seq + '.nii'
        else:
            nifti_name = self.fixed_path + '/' + self.fixed_subj + '/' + 'T1Pre' + '.nii'

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
        fixed_img = caster.Execute(fixed_img)
        if center:
            self.recenter_img_z(fixed_img)
        if extra is None:
            self.fixed_img = fixed_img
        else:
            self.fixed_extra = fixed_img

    def load_moving_nifti(self, cut):
        nifti_name = self.moving_path + '/' + self.moving_subj + '/' + self.moving_seq + '.nii'

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

        moving_img = img
        moving_img.GetPixelIDTypeAsString()
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkFloat32)
        moving_img = caster.Execute(moving_img)
        self.moving_img = moving_img[:, :, cut:]

    def load_moving(self, center):
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
        if center:
            self.recenter_img_z(self.moving_img, offset=False)

    def load_moving_mask_nifti(self, cut):
        nifti_name = self.moving_path + '/' + self.moving_subj + '/' + self.moving_seq + '_mask.nii'

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(nifti_name)
        img = reader.Execute()
        self.moving_mask = img[:, :, cut:]

    def load_moving_mask(self):
        # png_names = '/'.join([self.moving_path, self.moving_subj, self.moving_seq,
        #                      '/Ground/*png'])
        png_names = '/'.join([self.moving_path, self.moving_subj, 'T1DUAL',
                              '/Ground/*png'])
        seg_img_list = []
        for i, fn in enumerate(sorted(glob.glob(png_names))):
            tmp_seg_img = sitk.GetArrayFromImage(sitk.ReadImage(fn))
            # mask out all organs besides the liver (val = 80)
            tmp_seg_img = np.where((tmp_seg_img > 79) & (tmp_seg_img < 81), 80, 0)
            seg_img_list.append(tmp_seg_img)

        seg_img_array = np.zeros((len(seg_img_list), seg_img_list[0].shape[0],
                                  seg_img_list[0].shape[1]), dtype=np.float32)
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
        paff['AutomaticTransformInitialization'] = ['true']
        paff['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
        paff['NumberOfSamplesForExactGradient'] = ['100000']
        pbsp['NumberOfSamplesForExactGradient'] = ['100000']
        # paff['MaximumNumberOfSamplingAttempts'] = ['2']
        # pbsp['MaximumNumberOfSamplingAttempts'] = ['2']
        paff['NumberOfSpatialSamples'] = ['5000']
        pbsp['NumberOfSpatialSamples'] = ['5000']
        paff['NumberOfHistogramBins'] = ['32', '64', '128', '256']
        #pbsp['NumberOfHistogramBins'] = ['32', '32', '64', '128']
        paff['MaximumNumberOfIterations'] = ['512']
        pbsp['MaximumNumberOfIterations'] = ['256']
        # paff['NumberOfResolutions'] = ['4']
        pbsp['NumberOfResolutions'] = ['2']
        paff['GridSpacingSchedule'] = ['6', '4', '2', '1.000000']
        # pbsp['GridSpacingSchedule'] = ['6', '4', '2', '1.000000']
        pbsp['GridSpacingSchedule'] = ['4', '1.0']
        # pbsp['FinalGridSpacingInPhysicalUnits'] = ['40', '40', str(self.fixed_img.GetSpacing()[2])]
        # pbsp['FinalGridSpacingInPhysicalUnits'] = ['40', '40', '40']
        pbsp['FinalGridSpacingInPhysicalUnits'] = ['64','64','64']
        pbsp['FinalBSplineInterpolationOrder'] = ['2']
        # pbsp['Metric0Weight'] = ['0.01']
        # pbsp['Metric1Weight'] = ['0.1']
        # paff['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']
        # pbsp['FixedImagePyramid'] = ['FixedShrinkingImagePyramid']

        # attempting to use multiple fixed images at once
        # paff['Registration'] = ['MultiMetricMultiResolutionRegistration']
        # paff['FixedImagePyramid'] = ['FixedSmoothingImagePyramid', 'FixedSmoothingImagePyramid']
        # paff['ImageSampler'] = ['RandomCoordinate', 'RandomCoordinate']
        # paff['Metric'] = ['AdvancedMattesMutualInformation', 'AdvancedMattesMutualInformation']
        # pbsp['Metric'] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty',
        #                  'AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
        # pbsp['FixedImagePyramid'] = ['FixedSmoothingImagePyramid', 'FixedSmoothingImagePyramid']
        # pbsp['ImageSampler'] = ['RandomCoordinate', 'RandomCoordinate']
        #                         'RandomCoordinate', 'RandomCoordinate']
        paff['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
        pbsp['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']

        self.p_map_vector.append(paff)
        self.p_map_vector.append(pbsp)
        if self.verbose:
            sitk.PrintParameterMap(self.p_map_vector)

    def register_imgs(self, extra=False):
        self.elastixImageFilter = sitk.ElastixImageFilter()
        self.elastixImageFilter.SetFixedImage(self.fixed_img)
        if extra:
            self.elastixImageFilter.AddFixedImage(self.fixed_extra)

        self.elastixImageFilter.SetMovingImage(self.moving_img)
        self.elastixImageFilter.SetParameterMap(self.p_map_vector)
        self.elastixImageFilter.Execute()
        self.moving_img_result = self.elastixImageFilter.GetResultImage()
        self.moving_img_result.CopyInformation(self.fixed_img)

    def gen_mask(self, smooth=False):
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(
            self.elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetMovingImage(self.moving_mask)
        transformixImageFilter.Execute()
        self.moving_mask_result = transformixImageFilter.GetResultImage()

        if smooth:
            tmp_img = sitk.GetArrayFromImage(self.moving_mask_result)
            tmp_img = np.where((tmp_img > 50) & (tmp_img < 100), 80, 0)
            self.moving_mask_result = sitk.GetImageFromArray(tmp_img)

        self.moving_mask_result.CopyInformation(self.fixed_img)
        self.moving_mask_result = sitk.Cast(self.moving_mask_result, sitk.sitkFloat32)

    def recenter_img_z(self, sitk_img, offset=False):
        spacing = sitk_img.GetSpacing()[2]
        layers = sitk_img.GetSize()[2]
        orig = sitk_img.GetOrigin()
        if not offset:
            sitk_img.SetOrigin([orig[0], orig[1], spacing*(-layers/2)])
        else:
            sitk_img.SetOrigin([orig[0], orig[1], spacing*(-layers/1.5)])


def add_liver_mask(ds, moving_name='15', extra_name='extra1', snapmask=False):
    '''Generate a mask from the liver registration method, and place it into the given "extra" slot.
    Assumes you are using an xarray dataset from the MREDataset class.'''

    for sub in tqdm(ds.subject):
        if snapmask:
            if str(sub.values) in ['371', '396']:
                continue
            fixed_path = '/pghbio/dbmi/batmanlab/Data/MRE/'
            nifti_name = fixed_path + '/' + str(sub.values) + '/' + 'seg.nii'
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(nifti_name)
            mask = sitk.GetArrayFromImage(reader.Execute())
        else:

            mask_maker = MRELiverMask(str(sub.values), moving_name, verbose=True, center=True,
                                      fixed_seq='T1Pre', moving_seq='T1_inphase', cut=15)
            mask_maker.gen_param_map()
            mask_maker.register_imgs()
            mask_maker.gen_mask(smooth=True)
            mask = sitk.GetArrayFromImage(mask_maker.moving_mask_result)
        mask = np.where(mask >= 1, 1, 0)
        ds['image'].loc[dict(sequence=extra_name, subject=sub)] = mask

    new_sequence = [a.replace(extra_name, 'liverMsk') for a in ds.sequence.values]
    ds = ds.assign_coords(sequence=new_sequence)
    return ds
