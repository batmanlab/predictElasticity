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
import matplotlib.pyplot as plt

# Important DICOM Tags (https://www.dicomlibrary.com/dicom/dicom-tags/)
# '0008|0032': Acquision Time
# '0010|0030': Patient B-day
# '0018|0020': Scanning Sequence
# '0018|0021': Sequence Variant
# '0018|0022': Scan Options
# '0018|0023': MR Acquisition Type
# '0018|0050': Slice Thickness
# '0018|0080': Repetition Time
# '0018|0081': Echo Time
# '0018|0087': Mag Field Strength
# '0018|1060': Trigger Time (for splitting dicom series)


class MRIImage:
    '''Class for taking a NIFTI 3D MRI and converting it to an xarray object.'''

    def __init__(self, image, patient, sequence):
        npimage = sitk.GetArrayFromImage(image)

        x = np.full(image.GetSize()[0], image.GetSpacing()[0])
        x[0] = image.GetOrigin()[0]
        x = np.cumsum(x)

        y = np.full(image.GetSize()[1], image.GetSpacing()[1])
        y[0] = image.GetOrigin()[1]
        y = np.cumsum(y)[::-1]

        if image.GetDimension() == 3:
            z = np.full(image.GetSize()[2], image.GetSpacing()[2])
            z[0] = image.GetOrigin()[2]
            z = np.cumsum(z)
        else:
            npimage = np.asarray([npimage])
            z = [0]

        self.da = xr.DataArray(npimage, coords=[('z', z), ('y', y), ('x', x)])
        self.da.name = sequence
        self.da.attrs['patient'] = patient


class MREDataset:
    '''Wrapper of xarray.Dataset for loading MRE data.  Appropriately interpolates images.'''

    def __init__(self, data_path, dx, dy, dz, n_extras):
        self.data_path = data_path
        subjects = sorted([x for x in os.listdir(self.data_path) if x.isdigit()])
        self.subdir_names = ['SE00001', 'SE00002', 'SE00003', 'SE00004', 'SE00006', 'SE00005']
        self.sequence_labels = ['T1Pre', 'T1Pos', 'T2FR', 'T2SS', 'elast', 'elastMsk']
        self.extra_labels = [f'extra{n}' for n in range(n_extras)]

        dsub = len(subjects)
        dseq = len(self.sequence_labels)+len(self.extra_labels)

        self.ds = xr.Dataset({'image': (['subject', 'sequence', 'z', 'y', 'x'],
                                        np.zeros((dsub, dseq, dz, dy, dx), dtype=np.float32)),
                              'age': (['subject'], np.zeros(dsub, dtype=np.int8)),
                              'z_space': (['subject', 'sequence'],
                                          np.zeros((dsub, dseq), dtype=np.float16)),
                              'z_space_std': (['subject', 'sequence'],
                                              np.zeros((dsub, dseq), dtype=np.float16))
                              },

                             coords={'subject': subjects,
                                     'sequence': self.sequence_labels+self.extra_labels})

        self.gen_ref_image(dx, dy, dz)

    def gen_ref_image(self, dx, dy, dz):
        self.ref_image = sitk.GetImageFromArray(np.ones((dz, dy, dx), dtype=np.float32))

    def fill_ref_image(self, image):
        self.ref_image = sitk.Cast(self.ref_image, image.GetPixelIDValue())
        self.ref_image.SetSpacing((1.7, 1.7, image.GetSpacing()[-1]))
        self.ref_image.SetOrigin(image.GetOrigin())
        # self.ref_image.SetOrigin((0,0,0))
        self.ref_image.SetDirection(image.GetDirection())

    def load_data(self, norm=False, write_nifti=False, minimal=False):
        '''Load data into MREDataset'''

        # for subj in tqdm_notebook(self.ds.coords['subject'].values[6:7], desc='Subject'):
        for subj in tqdm_notebook(self.ds.coords['subject'].values, desc='Subject'):
            full_path = self.data_path + f'/{subj}/DICOM/ST00001'
            seq_holder_list = []
            elast_ref = None
            for i, sdir in enumerate(self.subdir_names):
                # Make a sequence holder so we can properly determine the name:

                seq_path = full_path + '/' + sdir
                seq_holder = SequenceHolder(*self.load_sequence(seq_path), subj)
                if sdir == 'SE00006':
                    seq_holder.seq_name = 'elast'
                elif sdir == 'SE00005':
                    seq_holder.seq_name = 'elastMsk'
                if i == 0:
                    self.fill_ref_image(seq_holder.image)
                seq_holder.gen_interp_image(self.ref_image, None, True)

                if not minimal:
                    if sdir not in ['SE00006', 'SE00005']:
                        seq_holder.clean_image_background()
                    seq_holder.gen_interp_image(self.ref_image, elast_ref)
                    if sdir == 'SE00006':
                        elast_ref = seq_holder.center_ref

                # seq_holder.gen_interp_image(self.ref_image)
                seq_holder_list.append(seq_holder)

            self.determine_seq_name(seq_holder_list)
            self.center_images_reg(seq_holder_list)
            self.assign_images(seq_holder_list, subj)

            if write_nifti:
                self.write_nifti(seq_holder_list, subj)
            self.ds['age'].loc[{'subject': subj}] = seq_holder_list[0].age

        self.ds = self.ds.assign_coords(x=range(len(self.ds.x)),
                                        y=range(0, -len(self.ds.y), -1),
                                        z=range(len(self.ds.z)))

    def write_data_netcdf(self, out_name):
        self.ds.to_netcdf(self.data_path+'/'+out_name)

    def load_sequence(self, path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
        reader.LoadPrivateTagsOn()  # Get DICOM Info
        image = reader.Execute()
        return image, reader

    def determine_seq_name(self, seq_holder_list):
        '''Assign correct names to sequences in place'''

        early_time = -1
        t1pre_index = t1pos_index = 99
        for i, seq_holder in enumerate(seq_holder_list):
            if seq_holder.seq_name is not None:
                continue

            scan_seq = seq_holder.reader.GetMetaData(0, '0018|0020')
            scan_opt = seq_holder.reader.GetMetaData(0, '0018|0022')
            acq_time = seq_holder.reader.GetMetaData(0, '0008|0032')
            date = seq_holder.reader.GetMetaData(0, '0008|0022')
            acq_time = datetime(int(date[0:4]), int(date[4:6]), int(date[6:]), int(acq_time[:2]),
                                int(acq_time[2:4]), int(acq_time[4:]))

            if scan_seq not in ['SE', 'GR']:
                raise ValueError('Scan seq (0018|0020) has value: {scan_seq}')

            if scan_seq == 'SE':
                if 'SS_GEMS' in scan_opt:
                    seq_holder.seq_name = 'T2SS'
                elif 'FC_SLICE_AX_GEMS' in scan_opt:
                    seq_holder.seq_name = 'T2FR'
                else:
                    raise ValueError(f'Scan opt (0018|0022) has value: {scan_opt}')

            elif scan_seq == 'GR':
                if early_time == -1:
                    t1pre_index = i
                    early_time = acq_time
                else:
                    t1pos_index = i
                    if acq_time > early_time:
                        continue
                    else:
                        t1pos_index, t1pre_index = t1pre_index, t1pos_index

        seq_holder_list[t1pos_index].seq_name = 'T1Pos'
        seq_holder_list[t1pre_index].seq_name = 'T1Pre'

    def assign_images(self, seq_holder_list, subj):
        for seq_holder in seq_holder_list:
            self.ds['image'].loc[{'sequence': seq_holder.seq_name,
                                  'subject': subj}] = seq_holder.np_image
            self.ds['z_space'].loc[{'sequence': seq_holder.seq_name,
                                    'subject': subj}] = seq_holder.spacing[-1]
            self.ds['z_space_std'].loc[{'sequence': seq_holder.seq_name,
                                        'subject': subj}] = np.std(seq_holder.spacing[-1])

    def recenter_img_z(self, sitk_img):
        spacing = sitk_img.GetSpacing()[2]
        layers = sitk_img.GetSize()[2]
        orig = sitk_img.GetOrigin()
        sitk_img.SetOrigin([orig[0], orig[1], spacing*(-layers/2)])

    def center_images_reg(self, seq_holder_list):
        for seq_holder in seq_holder_list:
            if seq_holder.seq_name == 'T2SS':
                fixed_img = seq_holder.image
            if seq_holder.seq_name == 'T1Pre':
                moving_img_1 = seq_holder.image
            elif seq_holder.seq_name == 'T1Pos':
                moving_img_2 = seq_holder.image
        moving_img_1.CopyInformation(fixed_img)
        moving_img_2.CopyInformation(fixed_img)

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_img)
        elastixImageFilter.SetMovingImage(moving_img_1)
        params = sitk.GetDefaultParameterMap("rigid")
        params['AutomaticTransformInitialization'] = ['true']
        params['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
        params['NumberOfSamplesForExactGradient'] = ['100000']
        params['NumberOfSpatialSamples'] = ['5000']
        params['NumberOfResolutions'] = ['1']
        params['NumberOfHistogramBins'] = ['128']
        params['GridSpacingSchedule'] = ['1.000000']
        elastixImageFilter.SetParameterMap(params)
        # sitk.PrintParameterMap(elastixImageFilter.GetParameterMap())
        # input()

        # elastixImageFilter.LogToFileOn()
        # elastixImageFilter.SetOutputDirectory('elastix_log')
        elastixImageFilter.Execute()
        moving_res_1 = elastixImageFilter.GetResultImage()
        moving_res_1.CopyInformation(moving_img_1)

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_img)
        elastixImageFilter.SetMovingImage(moving_img_2)
        params = sitk.GetDefaultParameterMap("rigid")
        params['AutomaticTransformInitialization'] = ['true']
        params['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
        params['NumberOfSamplesForExactGradient'] = ['100000']
        params['NumberOfSpatialSamples'] = ['5000']
        params['NumberOfResolutions'] = ['1']
        params['NumberOfHistogramBins'] = ['128']
        params['GridSpacingSchedule'] = ['1.000000']
        # params['AutomaticTransformInitialization'] = ['true']
        # params['AutomaticTransformInitializationMethod'] = ['GeometricalCenter']
        elastixImageFilter.SetParameterMap(params)
        elastixImageFilter.Execute()
        moving_res_2 = elastixImageFilter.GetResultImage()
        moving_res_2.CopyInformation(moving_img_2)

        for seq_holder in seq_holder_list:
            if seq_holder.seq_name == 'T1Pre':
                seq_holder.image = moving_res_1
            elif seq_holder.seq_name == 'T1Pos':
                seq_holder.image = moving_res_2

    def write_nifti(self, seq_holder_list, subj):
        for seq_holder in seq_holder_list:
            sitk.WriteImage(seq_holder.new_image,
                            self.data_path + f'/{subj}/{seq_holder.seq_name}.nii')


class SequenceHolder:
    '''Small data class for storing an image sequence'''

    def __init__(self, image, reader, subj):
        self.image = image
        self.reader = reader
        self.seq_name = None
        self.spacing = self.image.GetSpacing()
        self.age = self.reader.GetMetaData(0, '0010|0030')
        self.subj = subj
        # self.clean_image_background()

    def clean_image_background(self):
        fuzzy_image = sitk.GetArrayFromImage(self.image)
        for i in range(len(fuzzy_image)):
            mod_fuz = np.where(fuzzy_image[i] < 150, fuzzy_image[i], np.nan)
            mod_fuz = np.where(mod_fuz > 1, mod_fuz, np.nan)
            mean_val = np.nanmean(mod_fuz)
            elevation_map = sobel(fuzzy_image[i])
            markers = np.zeros_like(fuzzy_image[i])
            if self.subj in ['404']:
                markers[fuzzy_image[i] <= mean_val*2] = 1
                markers[fuzzy_image[i] > mean_val*3] = 2
            else:
                markers[fuzzy_image[i] <= mean_val*0.5] = 1
                markers[fuzzy_image[i] > mean_val*2] = 2
            segmentation = morphology.watershed(elevation_map, markers)
            segmentation = (segmentation-1).astype(bool)
            segmentation = morphology.remove_small_objects(segmentation, 15)
            segmentation = ndi.binary_closing(segmentation, np.ones((8, 8)))
            segmentation = ndi.binary_fill_holes(segmentation)
            # segmentation = ndi.binary_erosion(segmentation)
            if self.subj in ['396', '365', '404']:
                segmentation = morphology.remove_small_objects(segmentation, 200)
            else:
                segmentation = morphology.remove_small_objects(segmentation, 100)
            segmentation = morphology.convex_hull_image(segmentation)
            clean_img = np.where(segmentation, fuzzy_image[i], 0)
            fuzzy_image[i] = clean_img

        cleaned_sitk = sitk.GetImageFromArray(fuzzy_image)
        cleaned_sitk.CopyInformation(self.image)
        cleaned_sitk = sitk.Cast(cleaned_sitk, self.image.GetPixelIDValue())
        self.image = cleaned_sitk

    def gen_interp_image(self, ref_image, center_ref=None, first_pass=False):

        ref_image.SetSpacing((1.7, 1.7, self.spacing[-1]))
        if first_pass:
            center = sitk.CenteredTransformInitializer(
                ref_image, self.image, sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            self.image = sitk.Resample(self.image, ref_image,
                                       center,
                                       sitk.sitkNearestNeighbor)
            self.np_image = sitk.GetArrayFromImage(self.image)

        else:
            # bad, assumes "elast" always shows up before "elastMsk"
            if self.seq_name != 'elastMsk':
                self.center_ref = sitk.GetArrayFromImage(self.image)
                self.center_ref = np.where(self.center_ref > 1, 1, 0)
                self.center_ref = sitk.GetImageFromArray(self.center_ref)
                self.center_ref.CopyInformation(self.image)
                self.center_ref = sitk.Cast(self.center_ref, self.image.GetPixelIDValue())
            else:
                self.center_ref = center_ref
            center = sitk.CenteredTransformInitializer(
                ref_image, self.center_ref, sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
            new_image = sitk.Resample(self.image, ref_image, center,
                                      sitk.sitkNearestNeighbor)

            self.new_image = new_image
            self.np_image = sitk.GetArrayFromImage(new_image)


def make_nifti_atlas(path=None):
    if path is None:
        path = ('/pghbio/dbmi/batmanlab/Data/'
                'CombinedHealthyAbdominalOrganSegmentation/MR_data_batch1')
    for subj in ['1', '2', '3', '5', '8', '10', '13', '15', '19', '20']:
        for iseq, seq in enumerate(['T1DUAL', 'T1DUAL', 'T2SPIR']):
            dcm_names = '/'.join([path, subj, seq, '/DICOM_anon/*dcm'])
            print(f"Moving Image info for {dcm_names}")

            moving_img_list = []

            for i, fn in enumerate(sorted(glob.glob(dcm_names))):
                if iseq == 0:  # In phase T1
                    if i % 2 == 0:
                        continue
                elif iseq == 1:  # Out phase T1
                    if i % 2 != 0:
                        continue
                if i == 1 or i == 2:
                    # grab a single image to copy its metadata
                    tmp_moving_img = sitk.ReadImage(fn)
                moving_img_list.append(sitk.GetArrayFromImage(sitk.ReadImage(fn)))

            moving_img_array = np.zeros((len(moving_img_list), moving_img_list[0].shape[1],
                                         moving_img_list[0].shape[2]), dtype=np.float32)
            for i, img in enumerate(moving_img_list):
                moving_img_array[i, :, :] = img

            moving_img = sitk.GetImageFromArray(moving_img_array)
            moving_img.SetDirection(tmp_moving_img.GetDirection())
            moving_img.SetOrigin(tmp_moving_img.GetOrigin())
            moving_img.SetSpacing(tmp_moving_img.GetSpacing())
            size = moving_img.GetSize()
            dims = moving_img.GetSpacing()
            orig = moving_img.GetOrigin()

            print("Image size:", size[0], size[1], size[2])
            print("Image dims:", dims[0], dims[1], dims[2])
            print("Image orig:", orig[0], orig[1], orig[2])

            # recenter
            spacing = moving_img.GetSpacing()[2]
            layers = moving_img.GetSize()[2]
            orig = moving_img.GetOrigin()
            moving_img.SetOrigin([orig[0], orig[1], spacing*(-layers/2)])

            if iseq == 0:
                nifti_name = 'T1_inphase.nii'
            elif iseq == 1:
                nifti_name = 'T1_outphase.nii'
            elif iseq == 2:
                nifti_name = 'T2.nii'
            sitk.WriteImage(moving_img,
                            path + f'/{subj}/{nifti_name}')

            png_names = '/'.join([path, subj, seq,
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

            moving_mask = sitk.GetImageFromArray(seg_img_array)
            moving_mask.SetDirection(moving_img.GetDirection())
            moving_mask.SetOrigin(moving_img.GetOrigin())
            moving_mask.SetSpacing(moving_img.GetSpacing())

            if iseq == 0:
                nifti_name = 'T1_inphase_mask.nii'
            elif iseq == 1:
                nifti_name = 'T1_outphase_mask.nii'
            elif iseq == 2:
                nifti_name = 'T2_mask.nii'
            sitk.WriteImage(moving_mask,
                            path + f'/{subj}/{nifti_name}')


def make_nifti_atlas_v2(data_path=None, subdirs=None):
    if data_path is None:
        data_path = Path(
            '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/MR')
    if subdirs is None:
        subdirs = [1, 10, 13, 15, 19, 2, 20, 21, 22, 3, 31, 32, 33, 34, 36, 37, 38, 39, 5, 8]

    for subdir in tqdm_notebook(subdirs, desc='Patients'):
        subdir = str(subdir)
        for seq in ['t1_pre_in', 't1_pre_out', 't2']:
            if seq == 't1_pre_in':
                mr_path = Path(data_path, subdir, 'T1DUAL/DICOM_anon/InPhase')
                seg_path = Path(data_path, subdir, 'T1DUAL/Ground')
            elif seq == 't1_pre_out':
                mr_path = Path(data_path, subdir, 'T1DUAL/DICOM_anon/OutPhase')
                seg_path = Path(data_path, subdir, 'T1DUAL/Ground')
            elif seq == 't2':
                mr_path = Path(data_path, subdir, 'T2SPIR/DICOM_anon')
                seg_path = Path(data_path, subdir, 'T2SPIR/Ground')

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(mr_path))
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()

            png_list = sorted(list(seg_path.glob('*.png')))
            for i, png in enumerate(png_list):
                seg_img_slice = sitk.GetArrayFromImage(sitk.ReadImage(str(png)))
                if i == 0:
                    seg_img_array = np.zeros((len(png_list), seg_img_slice.shape[0],
                                              seg_img_slice.shape[1]), dtype=np.uint16)
                seg_img_array[i, :, :] = seg_img_slice

            # mask out all organs besides the liver (val = 63)
            seg_img_array = np.where((seg_img_array >= 55) & (seg_img_array <= 70), 63, 0)
            seg_img = sitk.GetImageFromArray(seg_img_array)
            seg_img.CopyInformation(image)

            mr_name = '_'.join([subdir.zfill(2), seq, 'MR', '.nii'])
            seg_name = '_'.join([subdir.zfill(2), seq, 'mask', '.nii'])
            patient_path = Path(data_path, 'NIFTI', subdir.zfill(2))
            patient_path.mkdir(exist_ok=True)

            mr_path = Path(patient_path, mr_name)
            seg_path = Path(patient_path, seg_name)
            sitk.WriteImage(image, str(mr_path))
            sitk.WriteImage(seg_img, str(seg_path))


def make_nifti_atlas_v3(do_MR=True, do_CT=True, subdirs=None):
    if do_MR:
        data_path_MR = Path(
            '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/MR')
    if do_CT:
        data_path_CT = Path(
            '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/CT')
    if not do_MR and not do_CT:
        print('You did nothing, great job.')
        return None

    outpath = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/NIFTI')

    full_ct_subdirs = [1, 10, 14, 16, 18, 19, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 5, 6, 8]
    full_mr_subdirs = [1, 10, 13, 15, 19, 2, 20, 21, 22, 3, 31, 32, 33, 34, 36, 37, 38, 39, 5, 8]
    if subdirs is None:
        if do_MR and do_CT:
            subdirs = list(set(full_ct_subdirs+full_mr_subdirs))
        elif do_MR:
            subdirs = full_mr_subdirs
        elif do_CT:
            subdirs = full_ct_subdirs

    for subdir in tqdm_notebook(subdirs, desc='Patients'):
        subdir = str(subdir)
        if do_MR:
            for seq in ['t1_pre_in', 't1_pre_out', 't2']:
                if seq == 't1_pre_in':
                    img_path = Path(data_path_MR, subdir, 'T1DUAL/DICOM_anon/InPhase')
                    seg_path = Path(data_path_MR, subdir, 'T1DUAL/Ground')
                elif seq == 't1_pre_out':
                    img_path = Path(data_path_MR, subdir, 'T1DUAL/DICOM_anon/OutPhase')
                    seg_path = Path(data_path_MR, subdir, 'T1DUAL/Ground')
                elif seq == 't2':
                    img_path = Path(data_path_MR, subdir, 'T2SPIR/DICOM_anon')
                    seg_path = Path(data_path_MR, subdir, 'T2SPIR/Ground')

                save_name = '_'.join([subdir.zfill(3), seq, 'MR'])

                process_chaos_dicoms(img_path, seg_path, subdir, save_name, outpath)

        if do_CT:
            img_path = Path(data_path_CT, subdir, 'DICOM_anon')
            seg_path = Path(data_path_CT, subdir, 'Ground')

            save_name = '_'.join(['1'+subdir.zfill(2), 'CT'])
            process_chaos_dicoms(img_path, seg_path, subdir, save_name, outpath)


def process_chaos_dicoms(img_path, seg_path, subdir, save_name, outpath):
    if not img_path.exists():
        print(img_path, 'does not exist.')
        print('Cannot make', save_name)
        return None

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(img_path))
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
    reader.LoadPrivateTagsOn()  # Get DICOM Info
    image = reader.Execute()

    if 'MR' in save_name:
        png_list = sorted(list(seg_path.glob('*.png')))
    else:
        png_list = sorted(list(seg_path.glob('liver*.png')), reverse=True)
    for i, png in enumerate(png_list):
        seg_img_slice = sitk.GetArrayFromImage(sitk.ReadImage(str(png)))
        if i == 0:
            seg_img_array = np.zeros((len(png_list), seg_img_slice.shape[0],
                                      seg_img_slice.shape[1]), dtype=np.uint16)
        seg_img_array[i, :, :] = seg_img_slice

    # mask out all organs besides the liver (val = 63)
    if 'MR' in save_name:
        seg_img_array = np.where((seg_img_array >= 55) & (seg_img_array <= 70), 63, 0)
    else:
        seg_img_array = np.where((seg_img_array != 0), 63, 0)
    seg_img = sitk.GetImageFromArray(seg_img_array)
    seg_img.CopyInformation(image)

    if 'MR' in save_name:
        patient_path = Path(outpath, subdir.zfill(3))
    else:
        patient_path = Path(outpath, '1'+subdir.zfill(2))
    patient_path.mkdir(exist_ok=True)

    img_path = Path(patient_path, save_name+'_img.nii')
    seg_path = Path(patient_path, save_name+'_mask.nii')
    sitk.WriteImage(image, str(img_path))
    sitk.WriteImage(seg_img, str(seg_path))


def dicom_to_pandas(data_path, subdirs):

    def process_index(k):
        return tuple(k.split("_"))

    s_list = []
    for subdir in tqdm_notebook(subdirs, desc='subdir'):
        semi_path = Path(data_path, subdir)
        for patient in tqdm_notebook(list(semi_path.iterdir()), desc='patient'):
            patient_path = Path(semi_path, patient, 'ST0')
            img_folders = sorted(list(patient_path.iterdir()), key=lambda a: int(a.stem[2:]))
            reader = sitk.ImageSeriesReader()
            for img_files in img_folders:
                dicom_names = reader.GetGDCMSeriesFileNames(str(img_files))
                dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
                reader.SetFileNames(dicom_names)
                reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
                reader.LoadPrivateTagsOn()  # Get DICOM Info
                try:
                    reader.Execute()
                except RuntimeError as e:
                    print(e)
                    continue
                pid = reader.GetMetaData(0, '0010|0010').strip()
                desc = img_files.stem
                desc = re.match(r'(\D*)(\d*)', desc, re.I)
                desc = ''.join([desc.groups()[0], desc.groups()[1].zfill(3)])
                for i, name in enumerate(dicom_names):
                    name = Path(name).stem
                    name = re.match(r'(\D*)(\d*)', name, re.I)
                    name = ''.join([name.groups()[0], name.groups()[1].zfill(3)])
                    index = []
                    vals = []
                    index.append('pathname')
                    pathname = '/'.join(patient.parts[-2:])
                    vals.append(pathname)
                    for k in reader.GetMetaDataKeys(i):
                        v = reader.GetMetaData(i, k)
                        if type(v) == str:
                            v = v.encode('utf-8', 'ignore').decode()
                        index.append(k)
                        vals.append(v)
                    s_list.append(pd.Series(index=index, data=vals, name=f'{pid}_{desc}_{name}'))
    df = pd.concat(s_list, axis=1).T
    df.index = pd.MultiIndex.from_tuples([process_index(k) for k, v in df.iterrows()])
    df.sort_index(inplace=True)
    return df


def dicom_to_nifti(data_path, subdirs):
    '''Code for determining which dicom to keep, and then save it as a nifti.'''

    for subdir in tqdm_notebook(subdirs, desc='subdir'):
        semi_path = Path(data_path, subdir)
        for patient in tqdm_notebook(list(semi_path.iterdir()), desc='patient'):
            patient_path = Path(semi_path, patient, 'ST0')
            # if 'PA5' not in str(patient_path):
            #     continue
            img_folders = sorted(list(patient_path.iterdir()), key=lambda a: int(a.stem[2:]))
            reader = sitk.ImageSeriesReader()

            sel_dict = dict(t1_pre_water=False,
                            t1_pre_fat=False,
                            t1_pre_in=False,
                            t1_pre_out=False,
                            t1_pos_300_water=False,
                            t1_pos_300_fat=False,
                            t1_pos_300_in=False,
                            t1_pos_300_out=False,
                            t1_pos_art_water=False,
                            t1_pos_art_fat=False,
                            t1_pos_art_in=False,
                            t1_pos_art_out=False,
                            t2=False,
                            mre_raw=False,
                            mre=False,
                            mre_mask=False)

            for i, img_files in enumerate(img_folders):
                dicom_names = reader.GetGDCMSeriesFileNames(str(img_files))
                dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
                reader.SetFileNames(dicom_names)
                reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
                reader.LoadPrivateTagsOn()  # Get DICOM Info
                try:
                    img = reader.Execute()
                    # Flip the image based on origin (sitk will not use origin info to sort slices)
                except RuntimeError as e:
                    print(e)
                    continue
                pid = reader.GetMetaData(0, '0010|0010').strip()
                desc = reader.GetMetaData(0, '0008|103e').strip().encode('utf-8',
                                                                         'ignore').decode().lower()
                # print(pid, desc, img.GetSize(), img.GetNumberOfComponentsPerPixel())

                patient_path = Path(data_path, 'NIFTI', pid)
                patient_path.mkdir(exist_ok=True)
                if i == 0:
                    Path(patient_path, '_'.join([data_path.stem, subdir, patient.stem])).touch()

                name = select_image(img, desc, sel_dict)
                if name:
                    if 'art' in name:
                        name_map = ['0', '70', '160']
                        img_list = dicom_split_sort(dicom_names, 'art')
                        for j, img in enumerate(img_list):
                            try:
                                sitk.WriteImage(
                                    img, str(patient_path) + '/' +
                                    name.replace('art', name_map[j]) + '.nii')
                            except IndexError:
                                sitk.WriteImage(
                                    img, str(patient_path) + '/' +
                                    name.replace('art', 'extra_art') + '.nii')

                    elif 'raw' in name:
                        img_wave, img_raw = split_image(img, reader, 'mre')
                        sitk.WriteImage(
                            img_wave, str(patient_path) +
                            '/' + name.replace('raw', 'wave') + '.nii')
                        sitk.WriteImage(
                            img_raw, str(patient_path) +
                            '/' + name + '.nii')
                    elif name == 'mre':
                        img, mre_info = scrape_mre(dicom_names)
                        sitk.WriteImage(img, str(patient_path) + '/' + name + '.nii')
                        with open(Path(patient_path, f'{name}.pkl'), 'wb') as f:
                            pkl.dump(mre_info, f)
                    elif name == 'mre_mask':
                        img, mre_info = scrape_mre(dicom_names)
                        sitk.WriteImage(img, str(patient_path) + '/' + name + '.nii')

                    else:
                        # img = orient_image(img, name)
                        img = dicom_split_sort(dicom_names, name)
                        sitk.WriteImage(img, str(patient_path) + '/' + name + '.nii')


def scrape_mre(dicom_names):
    '''MRE images are not evenly split.  This reads each slice and sorts them by z-value, then
    returns all those z-values along with the full image.'''

    mre_list = [(sitk.ReadImage(path).GetOrigin()[-1], path) for path in dicom_names]
    sorted_mre_list = sorted(mre_list, key=lambda x: x[0], reverse=False)
    mre_paths = []
    z_vals = []
    for i in sorted_mre_list:
        z_vals.append(i[0])
        mre_paths.append(i[1])

    return sitk.ReadImage(mre_paths), z_vals


def dicom_split_sort(dicom_names, name):
    '''sitk does not seem to be able to orient images correctly in some cases.  This reads in the
    dicom files slice by slice, grabbing the origin info from each slice, then sorting the slices
    based off of those origins.  Includes special cases for dicoms that consist of more than one
    type of sequence.'''

    if 'art' in name:
        # 'art' (arterial) files contain up to 3 different sets of images at 3 timesteps
        # after administering contrast. This function split each timestep into its own image.
        current_trigger = -999
        multi_z_path_pair = []
        for path in dicom_names:
            img_slice = sitk.ReadImage(path)
            trigger = img_slice.GetMetaData('0018|1060')  # Get trigger time
            if trigger != current_trigger:
                multi_z_path_pair.append([])
                current_trigger = trigger
            multi_z_path_pair[-1].append((img_slice.GetOrigin()[-1], path))
        img_list = []
        for i, pair in enumerate(multi_z_path_pair):
            _, sorted_dicom_names = zip(*sorted(pair, key=lambda x: x[0], reverse=False))
            img_list.append(sitk.ReadImage(sorted_dicom_names))

        return img_list
    elif 'raw' in name:
        pass
    else:
        z_path_pair = [(sitk.ReadImage(path).GetOrigin()[-1], path) for path in dicom_names]
        z_path_pair = sorted(z_path_pair, key=lambda x: x[0], reverse=False)
        _, sorted_dicom_names = zip(*z_path_pair)

        return sitk.ReadImage(sorted_dicom_names)


def split_image(img, reader, mode='art'):
    if mode == 'art':
        # Split the arterial/70 sec/2 min post-contrast images into 3 seperate images
        img_bounds = []
        current_trigger = -999
        for i in range(img.GetSize()[-1]):
            v = reader.GetMetaData(i, '0018|1060')  # Get trigger time
            if v != current_trigger:
                img_bounds.append([i, i+1])
                current_trigger = v
            else:
                img_bounds[-1][1] += 1

        if len(img_bounds) == 1:
            return img, None, None
        elif len(img_bounds) == 2:
            img1 = img[:, :, img_bounds[0][0]:img_bounds[0][1]]
            img2 = img[:, :, img_bounds[1][0]:img_bounds[1][1]]
            return img1, img2, None
        else:
            img1 = img[:, :, img_bounds[0][0]:img_bounds[0][1]]
            img2 = img[:, :, img_bounds[1][0]:img_bounds[1][1]]
            img3 = img[:, :, img_bounds[2][0]:img_bounds[2][1]]
            return img1, img2, img3

    elif mode == 'mre':
        # Split an mre-raw image into the raw and wave images.  Take the first timestamp for each
        # image type.
        meta_data_wave = {}
        meta_data_raw = {}
        for i in range(img.GetSize()[-1]):
            min_max_filter = sitk.MinimumMaximumImageFilter()
            min_max_filter.Execute(img[:, :, i])
            min_pixel = min_max_filter.GetMinimum()
            location = float(reader.GetMetaData(i, '0027|1041'))
            if min_pixel < -10:
                if location not in meta_data_wave:
                    meta_data_wave[location] = i
            else:
                if location not in meta_data_raw:
                    meta_data_raw[location] = i

        wave_slices = []
        for i, k in enumerate(sorted(list(meta_data_wave.keys()), reverse=False)):
            my_slice = img[:, :, meta_data_wave[k]]
            # Clean up origin to prevent join conflicts
            if i == 0:
                origin = my_slice.GetOrigin()
                direction = my_slice.GetDirection()
            else:
                my_slice.SetOrigin(origin)
                my_slice.SetDirection(direction)
            wave_slices.append(my_slice)
        img_wave = sitk.JoinSeries(wave_slices)

        raw_slices = []
        for i, k in enumerate(sorted(list(meta_data_raw.keys()), reverse=False)):
            my_slice = img[:, :, meta_data_raw[k]]
            # Clean up origin to prevent join conflicts
            if i == 0:
                origin = my_slice.GetOrigin()
                direction = my_slice.GetDirection()
            else:
                my_slice.SetOrigin(origin)
                my_slice.SetDirection(direction)
            raw_slices.append(my_slice)
        img_raw = sitk.JoinSeries(raw_slices)

        return img_wave, img_raw


def select_image(img, desc, sel_dict):
    if img.GetNumberOfComponentsPerPixel() == 3:
        return False
    elif 'cor' in desc:
        return False

    if not sel_dict['t1_pre_water'] and is_t1_pre_water(desc):
        sel_dict['t1_pre_water'] = desc
        return 't1_pre_water'

    if not sel_dict['t1_pre_fat'] and is_t1_pre_fat(desc):
        sel_dict['t1_pre_fat'] = desc
        return 't1_pre_fat'

    if not sel_dict['t1_pre_in'] and is_t1_pre_in(desc):
        sel_dict['t1_pre_in'] = desc
        return 't1_pre_in'

    if not sel_dict['t1_pre_out'] and is_t1_pre_out(desc):
        sel_dict['t1_pre_out'] = desc
        return 't1_pre_out'

    if not sel_dict['t1_pos_300_water'] and is_t1_pos_300_water(desc):
        sel_dict['t1_pos_300_water'] = desc
        return 't1_pos_300_water'

    if not sel_dict['t1_pos_300_fat'] and is_t1_pos_300_fat(desc):
        sel_dict['t1_pos_300_fat'] = desc
        return 't1_pos_300_fat'

    if not sel_dict['t1_pos_300_in'] and is_t1_pos_300_in(desc):
        sel_dict['t1_pos_300_in'] = desc
        return 't1_pos_300_in'

    if not sel_dict['t1_pos_300_out'] and is_t1_pos_300_out(desc):
        sel_dict['t1_pos_300_out'] = desc
        return 't1_pos_300_out'

    if not sel_dict['t1_pos_art_water'] and is_t1_pos_art_water(desc):
        sel_dict['t1_pos_art_water'] = desc
        return 't1_pos_art_water'

    if not sel_dict['t1_pos_art_fat'] and is_t1_pos_art_fat(desc):
        sel_dict['t1_pos_art_fat'] = desc
        return 't1_pos_art_fat'

    if not sel_dict['t1_pos_art_in'] and is_t1_pos_art_in(desc):
        sel_dict['t1_pos_art_in'] = desc
        return 't1_pos_art_in'

    if not sel_dict['t1_pos_art_out'] and is_t1_pos_art_out(desc):
        sel_dict['t1_pos_art_out'] = desc
        return 't1_pos_art_out'

    if not sel_dict['t2'] and is_t2(desc):
        sel_dict['t2'] = desc
        return 't2'

    if not sel_dict['mre_raw'] and is_mre_raw(desc):
        sel_dict['mre_raw'] = desc
        return 'mre_raw'

    if not sel_dict['mre'] and is_mre(desc):
        sel_dict['mre'] = desc
        return 'mre'

    if not sel_dict['mre_mask'] and is_mre_mask(desc):
        sel_dict['mre_mask'] = desc
        return 'mre_mask'

    return False


def is_t1_pre_water(desc):
    if 'lava' in desc:
        if 'water' in desc:
            if 'pre' in desc:
                return True
            elif 'min' in desc or '+c' in desc:
                return False
            else:
                return True
    return False


def is_t1_pre_fat(desc):
    if 'lava' in desc:
        if 'fat' in desc:
            if 'pre' in desc:
                return True
            elif 'min' in desc or '+c' in desc:
                return False
            else:
                return True
    return False


def is_t1_pre_in(desc):
    if 'lava' in desc:
        if 'inphase' in desc:
            if 'pre' in desc:
                return True
            elif 'min' in desc or '+c' in desc:
                return False
            else:
                return True
    return False


def is_t1_pre_out(desc):
    if 'lava' in desc:
        if 'outphase' in desc:
            if 'pre' in desc:
                return True
            elif 'min' in desc or '+c' in desc:
                return False
            else:
                return True
    return False


def is_t1_pos_300_water(desc):
    if 'lava' in desc:
        if 'water' in desc:
            if '5min' in desc:
                return True
    return False


def is_t1_pos_300_fat(desc):
    if 'lava' in desc:
        if 'fat' in desc:
            if '5min' in desc:
                return True
    return False


def is_t1_pos_300_in(desc):
    if 'lava' in desc:
        if 'inphase' in desc:
            if '5min' in desc:
                return True
    return False


def is_t1_pos_300_out(desc):
    if 'lava' in desc:
        if 'outphase' in desc:
            if '5min' in desc:
                return True
    return False


def is_t1_pos_art_water(desc):
    if 'lava' in desc:
        if 'water' in desc:
            if 'art' in desc:
                return True
    return False


def is_t1_pos_art_fat(desc):
    if 'lava' in desc:
        if 'fat' in desc:
            if 'art' in desc:
                return True
    return False


def is_t1_pos_art_in(desc):
    if 'lava' in desc:
        if 'inphase' in desc:
            if 'art' in desc:
                return True
    return False


def is_t1_pos_art_out(desc):
    if 'lava' in desc:
        if 'outphase' in desc:
            if 'art' in desc:
                return True
    return False


def is_t2(desc):
    if 't2' in desc and 'ssfse' in desc:
        return True
    return False


def is_mre_raw(desc):
    if 'mr touch' in desc:
        return True
    return False


def is_mre(desc):
    if 'elastogram' in desc and 'mask' not in desc:
        return True
    elif 'stgry' in desc and 'stgrym' not in desc:
        return True
    return False


def is_mre_mask(desc):
    if 'elastogram' in desc and 'mask' in desc:
        return True
    elif 'stgrym' in desc:
        return True
    return False


def make_xr_dataset_for_chaos(patients, nx, ny, nz, output_name):
    '''Given a list of patient IDs, make an xarray object from the niftis.
    Only minimal checks are done here, assumptions are:
        1. Patients exist
        2. All 3 default-view niftis exist for MR
        3. All 3 default-view masks exist for MR
        4. Nifti and mask exist for CT
        5. All nifti data is stored in the expected location
    All images must be resized before used.

    patients = ['001',  '003',  '008',  '013',  '019',  '021',  '031',  '033',  '036',  '038',
    '101', '105',  '108',  '114',  '118',  '121',  '123',  '125', '127',  '129', '002',  '005',
    '010',  '015',  '020',  '022',  '032',  '034',  '037',  '039',  '102',  '106',  '110', '116',
    '119',  '122',  '124',  '126', '128',  '130']
    '''

    data_dir = Path(
        '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/NIFTI')

    # Initialize empty ds
    n_seq = 4
    ds = init_new_ds(patients, n_seq, nx, ny, nz)

    for i, pat in enumerate(tqdm_notebook(patients, desc='Patients')):
        full_path = Path(data_dir, pat)
        img_files = list(full_path.iterdir())

        if pat[0] == '0':
            t1_in = get_image_match(img_files, 't1_pre_in_MR_img', pat, nx, ny, nz)
            t1_out = get_image_match(img_files, 't1_pre_out_MR_img', pat, nx, ny, nz)
            t2 = get_image_match(img_files, 't2_MR_img', pat, nx, ny, nz)
            t1_in_mask = get_image_match(img_files, 't1_pre_in_MR_mask', pat, nx, ny, nz)
            t1_out_mask = get_image_match(img_files, 't1_pre_out_MR_mask', pat, nx, ny, nz)
            t2_mask = get_image_match(img_files, 't2_MR_mask', pat, nx, ny, nz)

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

            ds['mr_ct_id'].loc[{'subject': pat}] = 1

        elif pat[0] == '1':
            ct = get_image_match(img_files, 'CT_img', pat, nx, ny, nz)
            ct_mask = get_image_match(img_files, 'CT_mask', pat, nx, ny, nz)

            ds['image'].loc[{'subject': pat, 'sequence': 'ct'}] = (
                sitk.GetArrayFromImage(ct).T)
            ds['mask'].loc[{'subject': pat, 'sequence': 'ct'}] = (
                sitk.GetArrayFromImage(ct_mask).T)
            ds['mr_ct_id'].loc[{'subject': pat}] = 2

    # return ds
    if ds is not None:
        print(f'Writing file disk...')
        output_name = Path(data_dir.parents[0], f'xarray_{output_name}.nc')
        ds.to_netcdf(output_name)
        return ds


def init_new_ds(subj_list, n_seq, nx, ny, nz):
    if len(subj_list) == 0:
        return None

    ds = xr.Dataset({'image': (['subject', 'sequence', 'x', 'y', 'z'],
                               np.zeros((len(subj_list), n_seq, nx, ny, nz), dtype=np.int16)),
                    'mask': (['subject', 'sequence', 'x', 'y', 'z'],
                             np.zeros((len(subj_list), n_seq, nx, ny, nz), dtype=np.int16)),
                     'mr_ct_id': (['subject'], np.zeros(len(subj_list), dtype=np.int16))
                     },

                    coords={'subject': subj_list,
                            'sequence': ['t1_in', 't1_out', 't2', 'ct'],
                            'x': range(nx),
                            'y': range(ny)[::-1],
                            'z': range(nz)
                            }
                    )
    return ds


def get_image_match(img_file_list, name, pat, nx, ny, nz, resample=True):
    for img_file in img_file_list:
        if name in img_file.stem:
            reader = sitk.ImageFileReader()
            reader.SetImageIO("NiftiImageIO")
            reader.SetFileName(str(img_file))
            img = reader.Execute()
            img = sitk.Cast(img, sitk.sitkInt16)
            if resample:
                init_size = img.GetSize()
                init_spacing = img.GetSpacing()
                xy_change = init_size[0]/nx
                z_change = init_size[-1]/nz
                ref_img = sitk.GetImageFromArray(np.ones((nz, ny, nx), dtype=np.int16))
                ref_img.SetSpacing((init_spacing[0]*xy_change, init_spacing[1]*xy_change,
                                   init_spacing[2]*z_change))
                ref_img.SetOrigin(img.GetOrigin())

                center = sitk.CenteredTransformInitializer(
                    ref_img, img, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY,
                )
                if 'mask' in name:
                    interp_method = sitk.sitkNearestNeighbor
                else:
                    interp_method = sitk.sitkLinear
                img = sitk.Resample(img, ref_img, center, interp_method)
            return img
    print(f'Could not find matching image for {pat}, {name}')
    return None
