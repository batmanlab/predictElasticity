#! usr/bin/env python
import os
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
                                          np.zeros((dsub, dseq), dtype=np.float16))
                              },

                             coords={'subject': subjects,
                                     'sequence': self.sequence_labels+self.extra_labels})

        self.gen_ref_image(dx, dy, dz)

    def gen_ref_image(self, dx, dy, dz):
        self.ref_image = sitk.GetImageFromArray(np.ones((dz, dy, dx), dtype=np.float32))

    def fill_ref_image(self, image):
        self.ref_image = sitk.Cast(self.ref_image, image.GetPixelIDValue())
        self.ref_image.SetSpacing((1.6, 1.6, image.GetSpacing()[-1]))
        # self.ref_image.SetOrigin(image.GetOrigin())
        # self.ref_image.SetOrigin((0,0,0))
        self.ref_image.SetDirection(image.GetDirection())

    def load_data(self, norm=False, write_nifti=False):
        '''Load data into MREDataset'''

        for subj in tqdm_notebook(self.ds.coords['subject'].values, desc='Subject'):
            full_path = self.data_path + f'/{subj}/DICOM/ST00001'
            seq_holder_list = []
            elast_ref = None
            for i, sdir in enumerate(self.subdir_names):
                # Make a sequence holder so we can properly determine the name:

                seq_path = full_path + '/' + sdir
                seq_holder = SequenceHolder(*self.load_sequence(seq_path))
                if sdir == 'SE00006':
                    seq_holder.seq_name = 'elast'
                elif sdir == 'SE00005':
                    seq_holder.seq_name = 'elastMsk'

                if sdir not in ['SE00006', 'SE00005']:
                    seq_holder.clean_image_background()
                if i == 0:
                    self.fill_ref_image(seq_holder.image)
                seq_holder.gen_interp_image(self.ref_image, elast_ref)
                if sdir == 'SE00006':
                    elast_ref = seq_holder.center_ref

                # seq_holder.gen_interp_image(self.ref_image)
                seq_holder_list.append(seq_holder)

            self.determine_seq_name(seq_holder_list)
            self.assign_images(seq_holder_list, subj)
            if write_nifti:
                self.write_nifti(seq_holder_list, subj)
            self.ds['age'].loc[{'subject': subj}] = seq_holder_list[0].age

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

    def write_nifti(self, seq_holder_list, subj):
        for seq_holder in seq_holder_list:
            sitk.WriteImage(seq_holder.new_image,
                            self.data_path + f'/{subj}/{seq_holder.seq_name}.nii')


class SequenceHolder:
    '''Small data class for storing an image sequence'''

    def __init__(self, image, reader):
        self.image = image
        self.reader = reader
        self.seq_name = None
        self.spacing = self.image.GetSpacing()
        self.age = self.reader.GetMetaData(0, '0010|0030')
        # self.clean_image_background()

    def clean_image_background(self):
        fuzzy_image = sitk.GetArrayFromImage(self.image)
        for i in range(len(fuzzy_image)):
            # plt.imshow(fuzzy_image[i])
            # plt.show()
            # edges = feature.canny(fuzzy_image[i, :, :], low_threshold=50)
            # plt.imshow(edges)
            # plt.show()
            # fill = ndi.binary_fill_holes(edges)
            # fill = morphology.remove_small_objects(fill)
            # if np.mean(fill) < 0.15:
            #     fill = ndi.binary_dilation(edges)
            #     fill = ndi.binary_closing(fill)
            #     fill = ndi.binary_fill_holes(fill)
            #     fill = morphology.remove_small_objects(fill)
            elevation_map = sobel(fuzzy_image[i])
            markers = np.zeros_like(fuzzy_image[i])
            markers[fuzzy_image[i] <= 75] = 1
            markers[fuzzy_image[i] > 75] = 2
            segmentation = morphology.watershed(elevation_map, markers)
            segmentation = ndi.binary_closing(segmentation-1, np.ones((1, 1)))
            segmentation = morphology.remove_small_objects(segmentation, 150)
            segmentation = ndi.binary_fill_holes(segmentation)
            segmentation = ndi.binary_erosion(segmentation)
            segmentation = morphology.remove_small_objects(segmentation, 100)
            segmentation = morphology.convex_hull_image(segmentation)
            clean_img = np.where(segmentation, fuzzy_image[i], 0)
            fuzzy_image[i] = clean_img

        cleaned_sitk = sitk.GetImageFromArray(fuzzy_image)
        cleaned_sitk.CopyInformation(self.image)
        cleaned_sitk = sitk.Cast(cleaned_sitk, self.image.GetPixelIDValue())
        self.image = cleaned_sitk

    def gen_interp_image(self, ref_image, center_ref=None):

        ref_image.SetSpacing((1.7, 1.7, self.spacing[-1]))
        # ref_image.SetOrigin((ref_image.GetOrigin()[0], ref_image.GetOrigin()[1],
        #                      self.image.GetOrigin()[2]))
        # ref_image.SetOrigin(self.image.GetOrigin())
        # ref_image.SetDirection(np.identity(3).flatten())
        # ref_image.SetDirection(self.image.GetDirection())

        # bad, assumes "elast" always shows up before "elastMsk"
        if self.seq_name != 'elastMsk':
            self.center_ref = sitk.GetArrayFromImage(self.image)
            self.center_ref = np.where(self.center_ref > 1, 1, 0)
            self.center_ref = sitk.GetImageFromArray(self.center_ref)
            self.center_ref.CopyInformation(self.image)
            self.center_ref = sitk.Cast(self.center_ref, self.image.GetPixelIDValue())
        else:
            self.center_ref = center_ref
        # pdb.set_trace()
        center = sitk.CenteredTransformInitializer(
            ref_image, self.center_ref, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        # center = sitk.CenteredTransformInitializer(
        #     ref_image, self.image, sitk.AffineTransform(3),
        #     sitk.CenteredTransformInitializerFilter.MOMENTS,
        # )
        # new_image = sitk.Resample(self.image, ref_image, sitk.Transform(3, sitk.sitkIdentity),
        #                           sitk.sitkNearestNeighbor)
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
