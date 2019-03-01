import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
from scipy import ndimage as ndi
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology


class MREDataset:
    '''Wrapper of xarray.Dataset for loading MRE data.  Appropriately interpolates images.'''

    def __init__(self, data_path, dx, dy, dz, n_extras):
        self.data_path = data_path
        subjects = sorted([x for x in os.listdir(self.data_path) if x.isdigit()])
        self.subdir_names = ['SE00001', 'SE00002', 'SE00003', 'SE00004', 'SE00005', 'SE00006']
        self.sequence_labels = ['T2SS', 'T1Pre', 'T1Pos', 'T2FR', 'elastMsk', 'elast']
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

        for subj in self.ds.coords['subject'].values:
            full_path = self.data_path + f'/{subj}/DICOM/ST00001'
            for i, sdir in enumerate(self.subdir_names):
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(full_path + '/' + sdir)
                reader.SetFileNames(dicom_names)
                reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
                reader.LoadPrivateTagsOn()  # Get DICOM Info

                image = reader.Execute()
                spacing = image.GetSpacing()

                self.ref_image.SetSpacing((1.5, 1.5, spacing[-1]))
                self.ref_image.SetOrigin((0, 0, 0))
                self.ref_image.SetDirection(np.identity(3).flatten())

                center = sitk.CenteredTransformInitializer(
                    self.ref_image, image, sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
                new_image = sitk.Resample(image, self.ref_image, center,
                                          sitk.sitkNearestNeighbor)

                np_image = sitk.GetArrayFromImage(new_image)
                if norm and sdir not in ['SE00005', 'SE00006']:
                    for im in range(4):
                        np_image[im, :, :] = (np_image[im, :, :] / np_image[im, :, :].max())

                self.ds['image'].loc[{'sequence': self.sequence_labels[i],
                                      'subject': subj}] = np_image
                self.ds['z_space'].loc[{'sequence': self.sequence_labels[i],
                                        'subject': subj}] = spacing[-1]

        self.ds['age'].loc[{'subject': subj}] = reader.GetMetaData(0, '0010|0030')

    def write_data(self, out_name):
        self.ds.to_netcdf(self.data_path+'/'+out_name)
