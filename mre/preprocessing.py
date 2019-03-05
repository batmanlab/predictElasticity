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
        self.subdir_names = ['SE00001', 'SE00002', 'SE00003', 'SE00004', 'SE00005', 'SE00006']
        self.sequence_labels = ['T1Pre', 'T1Pos', 'T2FR', 'T2SS', 'elastMsk', 'elast']
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
        self.ref_image = sitk.Image((dx, dy, dz), 2)

    def load_data(self, norm=False):
        '''Load data into MREDataset'''

        for subj in self.ds.coords['subject'].values:
            full_path = self.data_path + f'/{subj}/DICOM/ST00001'
            seq_holder_list = []
            for i, sdir in enumerate(self.subdir_names):
                # Make a sequence holder so we can properly determine the name:
                seq_path = full_path + '/' + sdir
                seq_holder = SequenceHolder(*self.load_sequence(seq_path))
                if sdir == 'SE00005':
                    seq_holder.seq_name = 'elastMsk'
                elif sdir == 'SE00006':
                    seq_holder.seq_name = 'elast'

                seq_holder.gen_interp_image(self.ref_image)
                seq_holder_list.append(seq_holder)

            self.determine_seq_name(seq_holder_list)
            self.assign_images(seq_holder_list, subj)
            self.ds['age'].loc[{'subject': subj}] = seq_holder_list[0].age

    def write_data(self, out_name):
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


class SequenceHolder:
    '''Small data class for storing an image sequence'''

    def __init__(self, image, reader):
        self.image = image
        self.reader = reader
        self.seq_name = None
        self.spacing = self.image.GetSpacing()
        self.age = self.reader.GetMetaData(0, '0010|0030')

    def gen_interp_image(self, ref_image):

        ref_image.SetSpacing((1.5, 1.5, self.spacing[-1]))
        ref_image.SetOrigin((0, 0, 0))
        ref_image.SetDirection(np.identity(3).flatten())

        center = sitk.CenteredTransformInitializer(
            ref_image, self.image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        new_image = sitk.Resample(self.image, ref_image, center,
                                  sitk.sitkNearestNeighbor)

        self.np_image = sitk.GetArrayFromImage(new_image)
