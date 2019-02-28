import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle as pkl
from scipy import ndimage as ndi
import SimpleITK as sitk
import skimage as skim
from skimage import feature, morphology


def add_elast_mask(ds, extra_name):
    '''Generate a mask from the elastMsk, and place it into the given "extra" slot.
    Assumes you are using an xarray dataset from the MREDataset class.'''

    for sub in list(ds.subject):
        for z in list(ds.z):
            # make initial mask from elast and elastMsk
            msk = ds.sel(sequence='elast', z=z, subject=sub).image.values -\
                ds.sel(sequence='elastMsk', z=z, subject=sub).image.values
            msk = np.where(msk > 1e-7, 1, 0)  # make mask high-contrast
            msk = morphology.binary_dilation(msk)  # fill in little holes

            # invert the mask so that 1s are in and 0s are out
            elast = ds.sel(sequence='elast', z=z, subject=sub).image.values
            msk = msk+np.where(elast < 1e-8, 1, 0)
            msk = 1-np.where(msk > 1, 1, msk)

            # place mask into 'extra' slot
            ds['image'].loc[dict(sequence=extra_name, z=z, subject=sub)] = msk

    # relabel the mask sequence
    new_sequence = [a.replace(extra_name, 'msk') for a in ds.sequence.values]
    ds = ds.assign_coords(sequence=new_sequence)
    return ds
