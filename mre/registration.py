import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
from datetime import datetime
from scipy import ndimage as ndi
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology
from skimage.morphology import binary_dilation
from skimage.restoration import inpaint
from skimage import color
import glob
from tqdm import tqdm


class RegPatient:
    '''Class that holds patient nifti images for registration.'''

    def __init__(self, subj, path):
        self.subj = subj
        self.path = path
        self.full_path = Path(path, subj)
        # print(self.full_path)
        self.images = {}
        self.load_niftis()

    def load_niftis(self):
        print(self.full_path)
        print('loading niftis')
        for f in self.full_path.glob('*.nii'):
            print(f)
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(str(f))
            img = reader.Execute()
            # if 'dwi' in str(f):
            #     img.SetOrigin((0, 0, 0))
            #     img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            if f.stem == 'wave':
                img = self.correct_wave_image(img)

            if f.stem == 'mre':
                print('mre', img.GetPixelIDTypeAsString())

            self.images[f.stem] = img

    def correct_wave_image(self, img):
        '''Convert wave image from rgb to gray, do inpainting'''
        print('correcting wave image')

        print(img.GetSize())
        print(img.GetOrigin())
        print(img.GetSpacing())
        print(img.GetDirection())
        print(img.GetNumberOfComponentsPerPixel())
        print(img.GetPixelIDTypeAsString())

        tmp_origin = img.GetOrigin()
        tmp_dir = img.GetDirection()
        tmp_spacing = img.GetSpacing()
        img_array = sitk.GetArrayViewFromImage(img)
        print(img_array.shape)
        # img_array = color.rgb2gray(img_array)
        print(img_array.shape)

        img_r = np.squeeze(img_array[:, :, :, 0]).astype(float)
        img_g = np.squeeze(img_array[:, :, :, 1]).astype(float)
        img_b = np.squeeze(img_array[:, :, :, 2]).astype(float)
        img_gr = np.where(img_r == 0, 0, img_g)
        img_gb = np.where(img_b == 0, 0, img_g)
        img_gg = np.where((img_b == 255) & (img_r == 255) & (img_g == 255), 1, 0)
        img_array = 0.001*(img_r+img_gr)  - 0.001*(img_b+img_gb) + img_gg

        array = np.ma.masked_where(img_array == 1, img_array)
        for i in range(img_array.shape[0]):
            array[i].mask = binary_dilation(array[i].mask)
            array[i].mask = binary_dilation(array[i].mask)
            img_array[i] = inpaint.inpaint_biharmonic(img_array[i], array[i].mask)
        img_array = 1000*img_array
        img_array = img_array.astype(np.int16)
        img = sitk.GetImageFromArray(img_array)
        img.SetOrigin(tmp_origin)
        img.SetDirection(tmp_dir)
        img.SetSpacing(tmp_spacing)
        print(img.GetSize())
        print(img.GetOrigin())
        print(img.GetSpacing())
        print(img.GetDirection())
        print(img.GetNumberOfComponentsPerPixel())
        print(img.GetPixelIDTypeAsString())
        return img


class Register:
    '''Class that registers a given fixed and moving image.'''

    def __init__(self, fixed_img, moving_img, verbose=True, dry_run=False, config=None,
                 fixed_mask=None, moving_mask=None):
        self.verbose = verbose
        self.fixed_img = fixed_img
        self.moving_img = moving_img
        self.config = config
        self.fixed_mask = fixed_mask
        self.moving_mask = moving_mask
        self.gen_param_map()
        if dry_run:
            self.moving_img_result = None
            return None
        else:
            self.register_imgs()

    def gen_param_map(self):
        '''Generate the parameter maps for registration.
        The `config` arg defines different sets used for different tasks.'''

        self.p_map_vector = sitk.VectorOfParameterMap()
        if self.config == 'mri_seq':
            paff = sitk.GetDefaultParameterMap("affine")
            paff['AutomaticTransformInitialization'] = ['true']
            paff['NumberOfSamplesForExactGradient'] = ['100000']
            paff['NumberOfSpatialSamples'] = ['5000']
            paff['NumberOfHistogramBins'] = ['64', '256', '512']
            paff['MaximumNumberOfIterations'] = ['256']
            paff['ImagePyramidRescaleSchedule'] = ['4', '2', '1']
            self.p_map_vector.append(paff)
        if self.config == 'dwi':
            paff = sitk.GetDefaultParameterMap("affine")
            paff['AutomaticTransformInitialization'] = ['true']
            # paff['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
            paff['NumberOfSamplesForExactGradient'] = ['100000']
            paff['NumberOfSpatialSamples'] = ['5000']
            paff['NumberOfHistogramBins'] = ['64', '256', '512']
            paff['MaximumNumberOfIterations'] = ['256']
            paff['ImagePyramidRescaleSchedule'] = ['4', '2', '1']
            self.p_map_vector.append(paff)
            self.p_map_vector.append(paff)
        elif self.config == 'mre_reg':
            paff = sitk.GetDefaultParameterMap("affine")
            self.p_map_vector.append(paff)
        elif self.config == 'mre_match':
            paff = sitk.GetDefaultParameterMap("rigid")
            paff['AutomaticTransformInitialization'] = ['true']
            paff['NumberOfSamplesForExactGradient'] = ['100000']
            paff['NumberOfSpatialSamples'] = ['40000']
            paff['NumberOfHistogramBins'] = ['128', '256']
            paff['MaximumNumberOfIterations'] = ['512']
            paff['NumberOfResolutions'] = ['2']
            # paff['ImageSampler'] = ['']
            paff['ImagePyramidRescaleSchedule'] = ['8', '8', '4',
                                                   '4', '4', '2']
            self.p_map_vector.append(paff)
        else:
            # paff = sitk.GetDefaultParameterMap("affine")
            paff = sitk.GetDefaultParameterMap("rigid")
            # pbsp = sitk.GetDefaultParameterMap("bspline")
            paff['AutomaticTransformInitialization'] = ['true']
            # paff['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
            paff['NumberOfSamplesForExactGradient'] = ['100000']
            # pbsp['NumberOfSamplesForExactGradient'] = ['100000']
            paff['NumberOfSpatialSamples'] = ['5000']
            # pbsp['NumberOfSpatialSamples'] = ['5000']
            # paff['NumberOfHistogramBins'] = ['32', '64', '256', '512']
            paff['NumberOfHistogramBins'] = ['64', '256', '512']
            paff['MaximumNumberOfIterations'] = ['256']
            # pbsp['MaximumNumberOfIterations'] = ['256']
            # pbsp['NumberOfResolutions'] = ['3']
            # paff['GridSpacingSchedule'] = ['6', '4', '2', '1.0']
            # pbsp['GridSpacingSchedule'] = ['6', '4', '2', '1.0']
            paff['GridSpacingSchedule'] = ['4', '2', '1.0']
            # pbsp['GridSpacingSchedule'] = ['4', '2', '1.0']
            # pbsp['FinalGridSpacingInPhysicalUnits'] = ['8', '8', '8']
            # pbsp['FinalBSplineInterpolationOrder'] = ['2']
            # paff['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
            # pbsp['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']

            self.p_map_vector.append(paff)
            # self.p_map_vector.append(pbsp)
        if self.verbose:
            sitk.PrintParameterMap(self.p_map_vector)

    def register_imgs(self, grid=False):
        self.elastixImageFilter = sitk.ElastixImageFilter()
        self.elastixImageFilter.SetFixedImage(self.fixed_img)

        self.elastixImageFilter.SetMovingImage(self.moving_img)
        if self.fixed_mask:
            self.elastixImageFilter.SetFixedMask(self.fixed_mask)
        if self.moving_mask:
            self.elastixImageFilter.SetMovingMask(self.moving_mask)
        self.elastixImageFilter.SetParameterMap(self.p_map_vector)
        self.elastixImageFilter.Execute()
        self.moving_img_result = self.elastixImageFilter.GetResultImage()
        # self.moving_img_result = sitk.RescaleIntensity(self.moving_img_result)
        np_image = sitk.GetArrayFromImage(self.moving_img_result)
        np_image = self.scale(np_image)
        self.moving_img_result = sitk.GetImageFromArray(np_image)
        self.moving_img_result.CopyInformation(self.fixed_img)
        self.moving_img_result = sitk.Cast(self.moving_img_result, self.moving_img.GetPixelID())

        if grid:
            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.SetTransformParameterMap(
                self.elastixImageFilter.GetTransformParameterMap())

            grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                                         size=self.moving_img.GetSize(),
                                         sigma=[5, 5], gridSpacing=[75.0, 75.0])
            grid_image.CopyInformation(self.moving_img)

            transformixImageFilter.SetMovingImage(grid_image)
            transformixImageFilter.Execute()
            self.grid_result = transformixImageFilter.GetResultImage()

    def scale(self, x, out_range=(0, 512)):
        domain = np.min(x), np.max(x)
        x = np.where(x == 0, np.nan, x)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
        return np.where(np.isnan(y), 0, y)
