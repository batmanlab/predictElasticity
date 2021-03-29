from functools import reduce
from pathlib import Path
from collections import OrderedDict
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, accuracy_score
from skimage import morphology
from scipy import ndimage as ndi
import pandas as pd
import xarray as xr
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
import SimpleITK as sitk
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize

import torch
from torch.utils.data import DataLoader

from mre.preprocessing import MRIImage
from mre.pytorch_arch_deeplab import DeepLabFeatures
from mre.mre_datasets import MRETorchDataset

hv.extension('bokeh')


def grid_plots(ds, rows, cols, title=None, xlabel=None, ylabel=None):
    '''helper function to automate the creation of grid plots'''
    n_rows = len(ds[rows])
    n_cols = len(ds[cols])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.75*n_rows),
                            sharex=True, sharey=True, squeeze=False)

    for i, row in enumerate(ds[rows]):
        for j, col in enumerate(ds[cols]):
            image = ds.sel(dict(zip([rows, cols], [row, col]))).image
            if j in [3, 4]:
                axs[i][j].imshow(image, vmin=0, vmax=9000)
            else:
                axs[i][j].imshow(image)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                axs[i][j].set_title(f'{col.values}', size=18)
            if j == 0:
                axs[i][j].set_ylabel(f'{row.values}', size=18)
    if title is None:
        title = f'Grid Plot of {rows} vs {cols}'
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    y_title = axs[0][0].get_position().get_points()[1][1]+(1/n_rows)*0.2
    fig.suptitle(title, size=22, y=y_title)
    return fig, axs


def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:, :, image_z] + alpha*moving[:, :, image_z]
    plt.figure()
    plt.imshow(sitk.GetArrayViewFromImage(img))
    plt.axis('off')
    plt.show()


def hv_pred_comp(targets, predictions, masks, names):
    targets = targets.data.cpu().numpy()
    predictions = predictions.data.cpu().numpy()
    masks = masks.data.cpu().numpy()

    print(targets.shape)
    ds_targets = xr.DataArray(targets,
                              dims=['subject', 'sequence', 'y', 'x'],
                              coords=[list(names), ['target'],
                                      range(0, -predictions.shape[2], -1),
                                      range(predictions.shape[3])
                                      ],
                              name='targets')
    ds_mask_liver = xr.DataArray(np.where(masks < 1e-6, np.nan, 1),
                                 dims=['subject', 'sequence', 'y', 'x'],
                                 coords=[list(names), ['mask_liver'],
                                         range(0, -predictions.shape[2], -1),
                                         range(predictions.shape[3])
                                         ],
                                 name='mask_liver')
    # ds_mask_conf = xr.DataArray(np.where(masks < 1, np.nan, 1),
    #                             dims=['subject', 'sequence', 'y', 'x'],
    #                             coords=[list(names), ['mask_conf'],
    #                                     range(0, -predictions.shape[2], -1),
    #                                     range(predictions.shape[3])
    #                                     ],
    #                             name='mask_liver')
    ds_predictions = xr.DataArray(predictions,
                                  dims=['subject', 'sequence', 'y', 'x'],
                                  coords=[list(names), ['prediction'],
                                          range(0, -predictions.shape[2], -1),
                                          range(predictions.shape[3])
                                          ],
                                  name='prediction')
    for subj in names:
        print(np.nanmean(ds_mask_liver.sel(subject=subj).values *
                         ds_targets.sel(subject=subj).values))
        print(np.nanmean(ds_mask_liver.sel(subject=subj).values *
                         ds_predictions.sel(subject=subj).values))


def hv_dl_vis_chaos(inputs, targets, names, seq_names=None, predictions=None):
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
        opts.Image(cmap='viridis', width=250, height=250, tools=['hover'], xaxis=None, yaxis=None),
        opts.Labels(text_color='white', text_font_size='8pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.Overlay(show_legend=False))

    if seq_names is None:
        seq_names = ['seq']

    inputs = inputs.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    print(inputs.shape)
    print(targets.shape)

    if inputs.ndim == 5:
        all_dims = ['subject', 'sequence', 'z', 'y', 'x']
        group_dims = ['subject', 'sequence', 'z']
        all_coords = [list(names), seq_names, range(inputs.shape[2]),
                      range(0, -inputs.shape[3], -1), range(inputs.shape[4])]
        print(all_coords)

    elif inputs.ndim == 4:
        all_dims = ['subject', 'sequence', 'y', 'x']
        group_dims = ['subject', 'sequence']
        all_coords = [list(names), seq_names,
                      range(0, -inputs.shape[2], -1), range(inputs.shape[3])]

    ds_inputs = xr.DataArray(inputs,
                             dims=all_dims,
                             coords=all_coords,
                             name='inputs')
    ds_targets = xr.DataArray(targets,
                              dims=all_dims,
                              coords=all_coords,
                              name='targets')

    hv_ds_inputs = hv.Dataset(ds_inputs.copy())
    hv_ds_targets = hv.Dataset(ds_targets.copy())

    hv_targets = hv_ds_targets.to(hv.Image, ['x', 'y'], groupby=group_dims, dynamic=True)
    hv_targets = hv_targets.redim.range(targets=(0, 1))

    hv_inputs = hv_ds_inputs.to(
        hv.Image, ['x', 'y'], groupby=group_dims, dynamic=True).redim.range(
            inputs=(-1, 1)).opts(cmap='viridis', title='Input')

    if predictions is not None:
        predictions = predictions.data.cpu().numpy()
        print(predictions.shape)
        ds_predictions = xr.DataArray(predictions,
                                      dims=all_dims,
                                      coords=all_coords,
                                      name='predictions')
        # hv_ds_predictions = [hv.Dataset(ds_predictions.sel(sequence=seq).copy()) for seq in
        #                      ds_predictions.sequence]
        hv_ds_predictions = hv.Dataset(ds_predictions.copy())
        hv_predictions = hv_ds_predictions.to(hv.Image, ['x', 'y'],
                                              groupby=group_dims, dynamic=True)
        pred_range = (ds_predictions.min(), ds_predictions.max())
        hv_predictions = hv_predictions.redim.range(predictions=pred_range)

        print('here')
        layout = hv.Layout(hv_inputs + hv_targets + hv_predictions).cols(3)
    else:
        layout = hv.Layout(hv_inputs + hv_targets).cols(3)
    return layout


def hv_dl_vis(inputs, targets, masks, names, predictions=None):
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
        opts.Image(cmap='viridis', width=350, height=350, tools=['hover'], xaxis=None, yaxis=None),
        opts.Labels(text_color='white', text_font_size='8pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.Overlay(show_legend=False))

    inputs = inputs.data.cpu().numpy()[:, 0:3, :, :]
    targets = targets.data.cpu().numpy()
    masks = masks.data.cpu().numpy()

    ds_inputs = xr.DataArray(inputs,
                             dims=['subject', 'sequence', 'y', 'x'],
                             coords=[list(names), ['T1Pre', 'T1Pos', 'T2SS'],
                                     range(0, -inputs.shape[2], -1),
                                     range(inputs.shape[3])
                                     ],
                             name='inputs')
    ds_targets = xr.DataArray(targets,
                              dims=['subject', 'sequence', 'y', 'x'],
                              coords=[list(names), ['elast'],
                                      range(0, -inputs.shape[2], -1),
                                      range(inputs.shape[3])
                                      ],
                              name='targets')
    ds_masks = xr.DataArray(masks,
                            dims=['subject', 'sequence', 'y', 'x'],
                            coords=[list(names), ['mask'],
                                    range(0, -inputs.shape[2], -1),
                                    range(inputs.shape[3])
                                    ],
                            name='masks')
    ds_masks = ds_masks.where(ds_masks < 1, 1)
    ds_masks = ds_masks.where(ds_masks > 0, np.nan)

    hv_ds_inputs = [hv.Dataset(ds_inputs.sel(sequence=seq).copy()) for seq in ds_inputs.sequence]
    hv_ds_targets = [hv.Dataset(ds_targets.sel(sequence=seq).copy()) for seq in ds_targets.sequence]
    hv_ds_masks = [hv.Dataset(ds_masks.sel(sequence=seq).copy()) for seq in ds_masks.sequence]

    slider = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name='mask')

    masks = hv_ds_masks[0].to(hv.Image, ['x', 'y'], groupby=['subject'],
                              dynamic=True).apply.opts(alpha=slider.param.value)
    masks.opts(cmap='Reds', tools=[])
    targets = hv_ds_targets[0].to(hv.Image, ['x', 'y'], groupby=['subject'], dynamic=True)
    targets = targets.redim.range(targets=(0, 80)).opts(title='Target')*masks

    input_list = [hv_ds.to(hv.Image, ['x', 'y'], groupby=['subject'],
                           dynamic=True).opts(cmap='viridis',
                                              title=f'Input {hv_ds.data.sequence.values}')*masks
                  for hv_ds in hv_ds_inputs]

    if predictions is not None:
        predictions = predictions.data.cpu().numpy()
        ds_prediction = xr.DataArray(predictions,
                                     dims=['subject', 'sequence', 'y', 'x'],
                                     coords=[list(names), ['prediction'],
                                             range(0, -predictions.shape[2], -1),
                                             range(predictions.shape[3])
                                             ],
                                     name='prediction')
        hv_ds_predictions = [hv.Dataset(ds_prediction.sel(sequence=seq).copy()) for seq in
                             ds_prediction.sequence]
        predictions = hv_ds_predictions[0].to(hv.Image, ['x', 'y'], groupby=['subject'],
                                              dynamic=True).redim.range(
                                                  prediction=(0, 80)).opts(title='Prediction')*masks

        layout = hv.Layout(input_list + [targets] + [predictions]).cols(3)
    else:
        layout = hv.Layout(input_list + [targets]).cols(3)
    return pn.Column(slider, layout)


def patient_series_viewer(path, patient, img_type='DICOM', info=''):
    '''Similar to pybreast viewer, but must natively handle a mix of 2d, 3d, scalar, and vector'''

    imopts = {'tools': ['hover', 'lasso_select'], 'width': 300, 'height': 300, 'cmap': 'viridis'}
    full_path = Path(path, patient)

    if img_type == 'NIFTI':
        img_folders = sorted(list(full_path.glob('*.nii')), key=lambda a: a.stem.split('_'))
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
    elif img_type == 'DICOM':
        img_folders = sorted(list(full_path.iterdir()), key=lambda a: int(a.stem[2:]))
        reader = sitk.ImageSeriesReader()
    elif img_type == 'DICOM_CHAOS':
        img_folders = [Path(full_path, 'T1DUAL/DICOM_anon/InPhase'),
                       Path(full_path, 'T1DUAL/DICOM_anon/OutPhase'),
                       Path(full_path, 'T2SPIR/DICOM_anon')]
        reader = sitk.ImageSeriesReader()
    elif img_type == 'DICOM_CHAOS_CT':
        img_folders = [Path(full_path, 'DICOM_anon')]
        reader = sitk.ImageSeriesReader()

    else:
        raise KeyError(f'img_type must be one of ["DICOM", "NIFTI"], got {img_type}')

    hv_images = []
    for img_files in img_folders:
        print(img_files)
        hvds_list = []
        if 'DICOM' in img_type:
            dicom_names = reader.GetGDCMSeriesFileNames(str(img_files))
            dicom_names = sorted(dicom_names, key=lambda a: Path(a).stem[2:].zfill(3))
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())
            desc = reader.GetMetaData(0, '0008|103e').strip().encode('utf-8', 'ignore').decode()
            pid = reader.GetMetaData(0, '0010|0010').strip()
            desc = ' '.join([img_files.stem, desc, pid])
            # if image.GetOrigin()[-1] > 0:
            #     image = image[:, :, ::-1]
        elif img_type == 'NIFTI':
            reader.SetFileName(str(img_files))
            desc = ' '.join(img_files.parts[-2:])
            image = reader.Execute()
            if 'dwi' in str(img_files):
                image = sitk.ConstantPad(image, (0, 0, 20), (0, 0, 0))
            print('direction', image.GetDirection())
            print('origin', image.GetOrigin())
            print('spacing', image.GetSpacing())

        npimg = sitk.GetArrayFromImage(image)
        print(npimg.shape)
        if npimg.shape[0] == 1:
            hv_images.append(hv.Image(npimg[0, :], label=desc).opts(**imopts))
        elif npimg.shape[-1] > 3:
            hvds_list.append(hv.Dataset(
                (np.arange(npimg.shape[2]), np.arange(npimg.shape[1])[::-1],
                 np.arange(npimg.shape[0]),
                 npimg), [f'x{desc}', f'y{desc}', f'z{desc}'],
                f'MRI{desc}'))
            print(hvds_list[-1])
            hv_images.append(hvds_list[-1].to(hv.Image, [f'x{desc}', f'y{desc}'],
                                              groupby=[f'z{desc}'],
                                              dynamic=True, label=desc).opts(**imopts,
                                                                             invert_yaxis=False))
        else:
            hv_images.append(hv.Image(npimg[0, :], label=desc).opts(**imopts))
        print()
    return hv.Layout(hv_images).opts(shared_axes=False, merge_tools=False, normalize=False,
                                     title=' '.join([patient, info])).cols(3)


def chaos_viewer(path, patient):
    '''Specifically for viewing CHAOS images with masks, assumes NIFTI'''

    clipping = {'min': 'transparent', 'max': 'red', 'NaN': 'transparent'}
    imopts = {'tools': ['hover'], 'width': 500, 'height': 500, 'cmap': 'viridis'}
    full_path = Path(path, patient)

    img_files = sorted(list(full_path.glob('*img*.nii')))
    mask_files = sorted(list(full_path.glob('*mask*.nii')))

    hv_images = []
    sliders = []
    csliders = []
    for f_img, f_mask in zip(img_files, mask_files):
        img_reader = sitk.ImageFileReader()
        img_reader.SetImageIO("NiftiImageIO")
        img_reader.SetFileName(str(f_img))
        mask_reader = sitk.ImageFileReader()
        mask_reader.SetImageIO("NiftiImageIO")
        mask_reader.SetFileName(str(f_mask))
        desc = ' '.join(f_img.parts[-2:])
        img = img_reader.Execute()
        mask = mask_reader.Execute()

        np_img = sitk.GetArrayFromImage(img)
        np_mask = sitk.GetArrayFromImage(mask)
        np_mask = np.where(np_mask < 1, np.nan, np_mask)

        ds_img = hv.Dataset(
            (np.arange(np_img.shape[2]), np.arange(np_img.shape[1]), np.arange(np_img.shape[0]),
             np_img), [f'x{desc}', f'y{desc}', f'z{desc}'],
            f'MRI{desc}')
        ds_mask = hv.Dataset(
            (np.arange(np_mask.shape[2]), np.arange(np_mask.shape[1]), np.arange(np_mask.shape[0]),
             np_mask), [f'x{desc}', f'y{desc}', f'z{desc}'],
            f'mask{desc}')
        slider = pn.widgets.FloatSlider(start=0, end=1, value=0.5, name=f'{desc}')
        if 'MR' in desc:
            cslider = pn.widgets.RangeSlider(start=0, end=2000, value=(0, 1000),
                                             name=f'contrast {desc}')
        else:
            cslider = pn.widgets.RangeSlider(start=-1000, end=2000, value=(-1000, 1000),
                                             name=f'contrast {desc}')
        sliders.append(slider)
        csliders.append(cslider)

        hv_img = ds_img.to(hv.Image, [f'x{desc}', f'y{desc}'],
                           groupby=[f'z{desc}'],
                           dynamic=True, label=desc).opts(**imopts, invert_yaxis=True)

        hv_mask = ds_mask.to(hv.Image, [f'x{desc}', f'y{desc}'],
                             groupby=[f'z{desc}'],
                             dynamic=True, label=desc).opts(cmap='Reds', invert_yaxis=True,
                                                            clipping_colors=clipping)

        hv_mask = hv_mask.apply.opts(alpha=slider.param.value)
        hv_img = hv_img.apply.opts(clim=cslider.param.value)
        vrange = {f'mask{desc}': (0, 10)}
        hv_mask = hv_mask.redim.range(**vrange)

        hv_images.append(hv_img * hv_mask)

    layout = hv.Layout(hv_images).opts(shared_axes=False, normalize=False,
                                       title=' '.join(patient)).cols(2)
    pn_layout = pn.pane.HoloViews(layout)
    wb = pn_layout.widget_box
    for s in sliders:
        wb.append(s)
    for c in csliders:
        wb.append(c)
    # wb.append(*sliders)
    # wb.append(cslider)
    # wb.append(cslider2)
    return pn.Row(wb, pn_layout)

    # return pn.Column(*sliders, hv.Layout(hv_images).opts(shared_axes=False, normalize=False,
    #                                                      title=' '.join(patient)).cols(2))


def patient_reg_comparison(fixed, moving_init, moving_final, grid=None):
    '''Comparing 3 images at once for alpha blending.  Expects sitk input format.'''

    imopts = {'tools': ['hover'], 'width': 300, 'height': 300}
    hvds_fixed = hv.Dataset(MRIImage(fixed, 'fixed', 'fixed').da)
    hvds_moving_init = hv.Dataset(MRIImage(moving_init, 'moving_init', 'moving_init').da)
    hvds_moving_init = hvds_moving_init.redim(z='z1')
    print(hvds_moving_init)
    if moving_final:
        hvds_moving_final = hv.Dataset(MRIImage(moving_final, 'moving_final', 'moving_final').da)

    hv_fixed = hvds_fixed.to(hv.Image, kdims=['x', 'y'], groupby=['z'], dynamic=True)
    hv_fixed.opts(**imopts, cmap='viridis', title='fixed and moving_final')

    hv_moving_init = hvds_moving_init.to(hv.Image, kdims=['x', 'y'], groupby=['z1'], dynamic=True)
    hv_moving_init.opts(**imopts, cmap='Reds', title='moving_init')

    if moving_final:
        hv_moving_final = hvds_moving_final.to(hv.Image, kdims=['x', 'y'],
                                               groupby=['z'], dynamic=True)
        hv_moving_final = hv_moving_final.redim.range(moving_final=(0.1, 256))
        hv_moving_final = hv_moving_final.opts(**imopts, cmap='Reds', title='moving_final',
                                               clipping_colors={'min': 'transparent'})
        slider2 = pn.widgets.FloatSlider(start=0, end=1, value=0.5, name='moving_final')
    if grid:
        hv_grid = hv.Image(sitk.GetArrayFromImage(grid), groupby=['z']).opts(**imopts,
                                                                             cmap='Greys_r')

    # Make an alpha slider
    slider1 = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name='moving_init')
    # Plot the slider and the overlayed images using the '*' operator
    if grid:
        return pn.Column(slider1, slider2,
                         rasterize(hv_fixed) *
                         rasterize(hv_moving_init.apply.opts(alpha=slider1.param.value)) *
                         rasterize(hv_moving_final.apply.opts(alpha=slider2.param.value)) +
                         rasterize(hv_grid))
    elif moving_final:
        return pn.Column(slider2,
                         hv_fixed *
                         hv_moving_final.apply.opts(alpha=slider2.param.value) +
                         hv_moving_init)
    else:
        return hv_fixed + hv_moving_init


def xr_viewer(xr_ds, grid_coords=None, group_coords=None, overlay_data='default', selection=None):
    '''generic xr ds viewer for pollack-format image holders'''
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                       fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12},
                       plot_size=300),
        opts.Image(cmap='viridis', width=550, height=550, tools=['hover'], xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='20pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.Overlay(show_legend=True))

    # Make holoviews dataset from xarray
    # xr_ds = xr_ds.sel(subject=['01', '03'])
    hv_ds = hv.Dataset(xr_ds)
    vdims = [v.name for v in hv_ds.vdims]
    kdims = [k.name for k in hv_ds.kdims]

    if grid_coords is None:
        if 'sequence' in kdims:
            grid_coords = 'sequence'

    if overlay_data == 'default':
        if 'mask' in vdims:
            overlay_data = 'mask'

    hv_ds_main_dict = {}
    for v in vdims:
        if (overlay_data is not None) and (v in overlay_data):
            continue
        else:
            hv_ds_main_dict[v] = hv_ds.to(hv.Image, kdims=['x', 'y'], vdims='image',
                                          dynamic=True)

    if overlay_data is not None:

        slider = pn.widgets.FloatSlider(start=0, end=1, value=0.5, name=overlay_data)
        hv_ds_over = hv_ds.to(hv.Image, kdims=['x', 'y'], vdims=overlay_data, dynamic=True)
        hv_ds_over = hv_ds_over.opts(cmap='Category10', clipping_colors={'min': 'transparent'},
                                     color_levels=10)
        redim = {overlay_data: (0.1, 256)}
        hv_ds_over = hv_ds_over.redim.range(**redim)
        hv_ds_over = hv_ds_over.apply.opts(alpha=slider.param.value)

        if len(hv_ds_main_dict) == 1:
            # layout = (hv_ds_main_dict[vdims[0]] * hv_ds_over)
            layout = (hv_ds_main_dict[vdims[0]] * hv_ds_over).grid('sequence')
            # layout = (hv_ds_main_dict[vdims[0]] * hv_ds_over)
            # layout = (hv_ds_over).grid('sequence', dynamic=True)
        else:
            layout = (hv_ds_main_dict[vdims[0]] * hv_ds_over).grid('sequence')
    else:
        layout = (hv_ds_main_dict[vdims[0]]).grid('sequence', dynamic=False)

    return pn.Column(slider, layout)


def xr_viewer_v2(xr_ds, grid_coords=None, group_coords=None,
                 overlay_data='default', selection=None, size=550, prediction=False, torch=False):
    '''generic xr ds viewer for pollack-format image holders, this time with support for the
    3d mre-style xarrays.  Some argument customization will be sacrificed for consistency and ease
    of use. Maybe.

    Valid vdims: image_mri, mask_mri, image_mre, mask_mre
    Valis kdims: subject, sequence, mask_type, x, y, z_mri, z_mre
    '''
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                       fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12},
                       plot_size=size),
        opts.Layout(fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12}),
        opts.Image(cmap='gray', width=size, height=size, xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='20pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.NdOverlay(show_legend=True, border_muted_alpha=0.1)
    )

    # Make holoviews dataset from xarray
    # xr_ds = xr_ds.sel(subject=['0006', '0384'])
    wave = True
    hv_ds_mri = hv.Dataset(xr_ds[['image_mri', 'mask_mri']])
    hv_ds_mre = hv.Dataset(xr_ds[['image_mre', 'mask_mre']])
    if prediction:
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre', 'mre_mask', 'mre_pred'])
        hv_ds_mre_2 = hv_ds_mre.select(mre_type=['wave', 'wave_pred'])
    elif not torch:
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre', 'mre_mask'])
        hv_ds_mre_2 = hv_ds_mre.select(mre_type=['mre_raw', 'wave'])
    elif torch and (len(xr_ds.mre_type) == 1):
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre'])
        wave = False
    else:
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre'])
        hv_ds_mre_2 = hv_ds_mre.select(mre_type=['wave'])
    print(hv_ds_mri)
    print(hv_ds_mre)

    hv_ds_mri_image = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='image_mri', dynamic=True)
    hv_ds_mri_mask = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='mask_mri',
                                  dynamic=True).opts(tools=[])

    hv_ds_mre_image_1 = hv_ds_mre_1.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                       dynamic=True).opts(cmap='viridis')
    hv_ds_mre_mask_1 = hv_ds_mre_1.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                      dynamic=True).opts(tools=[])
    if wave:
        hv_ds_mre_image_2 = hv_ds_mre_2.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                           dynamic=True).opts(cmap='viridis')
        hv_ds_mre_mask_2 = hv_ds_mre_2.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                          dynamic=True).opts(tools=[])

    slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='mask transparency')
    if torch:
        cslider = pn.widgets.RangeSlider(start=-2, end=2, value=(-1, 1), name='contrast')
        cslider2 = pn.widgets.RangeSlider(start=0, end=120, value=(0, 100), name='mre contrast')
    else:
        cslider = pn.widgets.RangeSlider(start=0, end=2000, value=(0, 1000), name='contrast')
        cslider2 = pn.widgets.RangeSlider(start=0, end=12000, value=(0, 10000), name='mre contrast')

    redim_image_mri = {'image_mri': (0, 1200)}
    hv_ds_mri_image = hv_ds_mri_image.redim.range(**redim_image_mri).opts(tools=['hover'])
    hv_ds_mri_image = hv_ds_mri_image.apply.opts(clim=cslider.param.value)
    redim_mask_mri = {'mask_mri': (0.1, 2)}
    hv_ds_mri_mask = hv_ds_mri_mask.opts(cmap='Category10', clipping_colors={'min': 'transparent'},
                                         color_levels=10)
    hv_ds_mri_mask = hv_ds_mri_mask.redim.range(**redim_mask_mri)
    hv_ds_mri_mask = hv_ds_mri_mask.apply.opts(alpha=slider.param.value)

    redim_image_mre_1 = {'image_mre_1': (0, 10000)}
    # hv_ds_mre_image_1 = hv_ds_mre_image_1.redim(image_mre='image_mre_1')
    hv_ds_mre_image_1 = hv_ds_mre_image_1.apply.opts(clim=cslider2.param.value)
    hv_ds_mre_image_1 = hv_ds_mre_image_1.redim.range(**redim_image_mre_1).opts(tools=['hover'])
    redim_mask_mre = {'mask_mre': (0.1, 2)}
    hv_ds_mre_mask_1 = hv_ds_mre_mask_1.opts(cmap='Category10',
                                             clipping_colors={'min': 'transparent'},
                                             color_levels=10)
    hv_ds_mre_mask_1 = hv_ds_mre_mask_1.redim.range(**redim_mask_mre)
    hv_ds_mre_mask_1 = hv_ds_mre_mask_1.apply.opts(alpha=slider.param.value)

    if wave:
        if torch:
            redim_image_mre_2 = {'image_mre': (-10, 10)}
            cslider3 = pn.widgets.RangeSlider(start=-20, end=20, value=(-10, 10),
                                              name='wave contrast')
        else:
            redim_image_mre_2 = {'image_mre': (-200, 200)}
            cslider3 = pn.widgets.RangeSlider(start=-2000, end=2000, value=(-200, 200),
                                              name='wave contrast')
        hv_ds_mre_image_2 = hv_ds_mre_image_2.apply.opts(clim=cslider3.param.value)
        hv_ds_mre_image_2 = hv_ds_mre_image_2.redim.range(**redim_image_mre_2).opts(tools=['hover'])
        redim_mask_mre = {'mask_mre': (0.1, 2)}
        hv_ds_mre_mask_2 = hv_ds_mre_mask_2.opts(cmap='Category10',
                                                 clipping_colors={'min': 'transparent'},
                                                 color_levels=10)
        hv_ds_mre_mask_2 = hv_ds_mre_mask_2.redim.range(**redim_mask_mre)
        hv_ds_mre_mask_2 = hv_ds_mre_mask_2.apply.opts(alpha=slider.param.value)
        layout = (((hv_ds_mre_image_1 * hv_ds_mre_mask_1).grid('mre_type') +
                  (hv_ds_mre_image_2 * hv_ds_mre_mask_2).grid('mre_type')) +
                  (hv_ds_mri_image * hv_ds_mri_mask).layout('sequence').cols(3)
                  ).cols(2)
    else:
        layout = (((hv_ds_mre_image_1 * hv_ds_mre_mask_1).grid('mre_type')) +
                  (hv_ds_mri_image * hv_ds_mri_mask).layout('sequence').cols(3)
                  ).cols(1)
    pn_layout = pn.pane.HoloViews(layout)
    wb = pn_layout.widget_box
    wb.append(slider)
    wb.append(cslider)
    wb.append(cslider2)
    if wave:
        wb.append(cslider3)

    # return pn.Column(slider, cslider2, layout, cslider)
    return pn.Column(wb, pn_layout)
    # return hv_ds_mri_image
    # return hv_ds_mre_image


def xr_viewer_chaos(xr_ds, grid_coords=None, group_coords=None,
                    overlay_data='default', selection=None, size=400, prediction=False,
                    torch=False):
    '''generic xr ds viewer for pollack-format image holders, this time with support for the
    3d chaos-style xarrays.  Some argument customization will be sacrificed for consistency and ease
    of use. Maybe.

    Valid vdims: image, mask,
    Valis kdims: subject, sequence, x, y, z
    '''
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                       fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12},
                       plot_size=size),
        opts.Layout(fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12}),
        opts.Image(cmap='gray', width=size, height=size, xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='20pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.NdOverlay(show_legend=True, border_muted_alpha=0.1)
    )

    # Make holoviews dataset from xarray
    # xr_ds = xr_ds.sel(subject=['0006', '0384'])
    hv_ds = hv.Dataset(xr_ds[['image', 'mask']])

    hv_ds_image = hv_ds.to(hv.Image, kdims=['x', 'y'], vdims='image', dynamic=True)
    hv_ds_mask = hv_ds.to(hv.Image, kdims=['x', 'y'], vdims='mask',
                          dynamic=True).opts(tools=[])

    slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='mask transparency')

    cslider = pn.widgets.RangeSlider(start=0, end=2000, value=(0, 1000), name='contrast')

    redim_image = {'image': (0, 1200)}
    hv_ds_image = hv_ds_image.redim.range(**redim_image).opts(tools=['hover'])
    hv_ds_image = hv_ds_image.apply.opts(clim=cslider.param.value)
    redim_mask = {'mask': (0.1, 2)}
    hv_ds_mask = hv_ds_mask.opts(cmap='Category10', clipping_colors={'min': 'transparent'},
                                 color_levels=10)
    hv_ds_mask = hv_ds_mask.redim.range(**redim_mask)
    hv_ds_mask = hv_ds_mask.apply.opts(alpha=slider.param.value)

    layout = (hv_ds_image * hv_ds_mask).grid('sequence')
    pn_layout = pn.pane.HoloViews(layout)
    wb = pn_layout.widget_box
    wb.append(slider)
    wb.append(cslider)

    # return pn.Column(slider, cslider2, layout, cslider)
    return pn.Column(wb, pn_layout)
    # return hv_ds_mri_image
    # return hv_ds_mre_image


def miccai_plots(ds, do_cor=True, save_name='test', erode=0):
    true_pixel = []
    pred_pixel = []
    true_subj = []
    pred_subj = []
    # if do_cor:
    #     slope = np.mean(ds['val_slope'].values)
    #     intercept = np.mean(ds['val_intercept'].values)
    #     print(slope, intercept)
    for subj in ds.subject:
        mask = ds.sel(subject=subj, mask_type='combo')['mask_mre'].values
        if erode != 0:
            for i in range(mask.shape[-1]):
                mask[:, :, i] = ndi.binary_erosion(mask[:, :, i],
                                                   iterations=erode).astype(mask.dtype)
        mask = np.where(mask > 0, mask, np.nan)
        true_mre_region = (ds.sel(subject=subj, mre_type='mre')['image_mre'].values * mask)
        true_mre_region = true_mre_region.flatten()
        true_mre_region = true_mre_region[~np.isnan(true_mre_region)]
        pred_mre_region = (ds.sel(subject=subj, mre_type='mre_pred')['image_mre'].values * mask)
        pred_mre_region = pred_mre_region.flatten()
        pred_mre_region = pred_mre_region[~np.isnan(pred_mre_region)]
        if do_cor:
            slope = np.mean(ds.sel(subject=subj)['val_slope'].values)
            intercept = np.mean(ds.sel(subject=subj)['val_intercept'].values)
            # print(slope, intercept)
            pred_mre_region = (pred_mre_region-intercept)/slope
            pred_mre_region = np.where(pred_mre_region > 0, pred_mre_region, 0)
        true_pixel.append(true_mre_region/1000)
        pred_pixel.append(pred_mre_region/1000)
        true_subj.append(np.nanmean(true_mre_region)/1000)
        pred_subj.append(np.nan_to_num(np.nanmean(pred_mre_region/1000)))

    true_pixel = np.concatenate(true_pixel)
    pred_pixel = np.concatenate(pred_pixel)
    df_subj = pd.DataFrame({'true': true_subj, 'predict': pred_subj, 'subject': ds.subject.values})
    df_subj['fibrosis'] = np.where(df_subj.true > 4, 'Severe Fibrosis', 'Mild Fibrosis')

    # Subj Plot
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(df_subj['predict'], params, x=df_subj['true'])
    fig, ax = plt.subplots()
    ymax = df_subj['predict'].max()+1
    xmax = df_subj['true'].max()+1
    rect1 = plt.Rectangle((0, 0), 4, 4, color='grey', alpha=0.3, label='Correct Classification')
    rect2 = plt.Rectangle((4, 4), xmax-4, ymax-4, color='grey', alpha=0.3)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    df_subj.query('true>4').plot(x='true', y='predict', kind='scatter',
                                 xlim=(0, xmax), ylim=(0, ymax), color='C1', ax=ax,
                                 label='High Stiffness')
    df_subj.query('true<4').plot(x='true', y='predict', kind='scatter',
                                 xlim=(0, xmax), ylim=(0, ymax), color='C0', ax=ax,
                                 label='Low Stiffness')
    plt.plot(df_subj['true'], result.best_fit, label='Best Fit', linewidth=2, c='k')
    plt.title('Predicted vs True Stiffness, Subject-wise', size=22)
    # plt.ylim(-1, df_subj['predict'].max())
    # plt.xlim(-1, df_subj['predict'].max())
    plt.xlabel('True Stiffness (kPa)', size=20)
    plt.ylabel('Predicted Stiffness (kPa)', size=20)
    lgnd = plt.legend(fontsize=15, frameon=False)
    for handle in lgnd.legendHandles:
        try:
            handle.set_sizes([60])
        except AttributeError:
            pass
    ax.tick_params(labelsize=16)
    # return result

    print(result.fit_report())
    print('R2:', 1 - result.residual.var() / np.var(df_subj['predict']))
    plt.savefig(f'../plots/subj_results_{save_name}.pdf', bbox_inches='tight')

    # Pixel Plot
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(pred_pixel, params, x=true_pixel)
    fig, ax = plt.subplots()
    hb = plt.hexbin(true_pixel, pred_pixel, bins='log', cmap='Blues')
    plt.plot(true_pixel, result.best_fit, label='Best Fit', linewidth=2, c='k')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log(# of Pixels)', fontsize=20)
    cb.ax.tick_params(labelsize=16)

    plt.title('Predicted vs True Stiffness, Pixel-wise', size=22)
    plt.ylim(0, pred_pixel.max())
    plt.xlim(0, pred_pixel.max())
    plt.xlabel('True Stiffness (kPa)', size=20)
    plt.ylabel('Predicted Stiffness (kPa)', size=20)
    ax.tick_params(labelsize=16)
    # return result

    print(result.fit_report())
    print('R2:', 1 - result.residual.var() / np.var(pred_pixel))
    plt.savefig(f'../plots/pixel_results_{save_name}.pdf', bbox_inches='tight')

    # Sensitivity S: TP/(TP+FN)
    tp = df_subj.query('true>4 and predict>4').count()[0]
    fn = df_subj.query('true>4 and predict<4').count()[0]
    print('tp', tp)
    print('fn', fn)
    print('sensitivity', tp/(tp+fn))

    # Specificity: TN/(TN+FP)
    tn = df_subj.query('true<4 and predict<4').count()[0]
    fp = df_subj.query('true<4 and predict>4').count()[0]
    print('tn', tn)
    print('fp', fp)
    print('specificity', tn/(tn+fp))

    print('accuracy', (tn+tp)/(tn+fp+tp+fn))
    return df_subj


def xr_viewer_models(xr_ds, size=250, do_cor=False):
    '''generic xr ds viewer for pollack-format image holders, this time with support for the
    3d mre-style xarrays.  Some argument customization will be sacrificed for consistency and ease
    of use. Maybe.

    Valid vdims: image_mri, mask_mri, image_mre, mask_mre
    Valis kdims: subject, sequence, mask_type, x, y, z_mri, z_mre
    '''
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                       fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12},
                       plot_size=size),
        opts.Layout(fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12}),
        opts.Image(cmap='gray', width=size, height=size, xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='20pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.NdOverlay(show_legend=True, border_muted_alpha=0.1)
    )

    # Make holoviews dataset from xarray
    hv_ds_mri = hv.Dataset(xr_ds[['image_mri', 'mask_mri']])
    if do_cor:
        cor = (xr_ds['image_mre']-xr_ds['val_intercept'])/xr_ds['val_slope']
        cor = np.where(cor > 0, cor, 0)
        xr_ds['image_mre'].loc[{}] = cor
        # xr_ds['image_mre'].loc[{}] = xr_ds['image_mre']+1000
    hv_ds_mre = hv.Dataset(xr_ds[['image_mre', 'mask_mre']])

    mre_types = list(xr_ds.mre_type.values)
    mre_types = [mre for mre in mre_types if mre not in ['mre_raw', 'mre', 'mre_pred', 'mre_wave']]
    hv_ds_mre = hv_ds_mre.select(mre_type=mre_types)
    print(hv_ds_mri)
    print(hv_ds_mre)

    hv_ds_mri_image = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='image_mri', dynamic=True)
    hv_ds_mri_mask = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='mask_mri',
                                  dynamic=True).opts(tools=[])

    hv_ds_mre_image = hv_ds_mre.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                   dynamic=True).opts(cmap='viridis')
    hv_ds_mre_mask = hv_ds_mre.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                  dynamic=True).opts(tools=[])

    slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='mask transparency')
    cslider = pn.widgets.RangeSlider(start=0, end=2000, value=(0, 1000), name='contrast')
    cslider2 = pn.widgets.RangeSlider(start=0, end=12000, value=(0, 10000), name='mre contrast')

    redim_image_mri = {'image_mri': (0, 1200)}
    hv_ds_mri_image = hv_ds_mri_image.redim.range(**redim_image_mri).opts(tools=['hover'])
    hv_ds_mri_image = hv_ds_mri_image.apply.opts(clim=cslider.param.value)
    redim_mask_mri = {'mask_mri': (0.1, 2)}
    hv_ds_mri_mask = hv_ds_mri_mask.opts(cmap='Reds', clipping_colors={'min': 'transparent'},
                                         color_levels=10)
    hv_ds_mri_mask = hv_ds_mri_mask.redim.range(**redim_mask_mri)
    hv_ds_mri_mask = hv_ds_mri_mask.apply.opts(alpha=slider.param.value)

    redim_image_mre = {'image_mre': (0, 10000)}
    # hv_ds_mre_image = hv_ds_mre_image.redim(image_mre='image_mre')
    hv_ds_mre_image = hv_ds_mre_image.apply.opts(clim=cslider2.param.value)
    hv_ds_mre_image = hv_ds_mre_image.redim.range(**redim_image_mre).opts(tools=['hover'])
    redim_mask_mre = {'mask_mre': (0.1, 2)}
    hv_ds_mre_mask = hv_ds_mre_mask.opts(cmap='Reds',
                                         clipping_colors={'min': 'transparent'},
                                         color_levels=10)
    hv_ds_mre_mask = hv_ds_mre_mask.redim.range(**redim_mask_mre)
    hv_ds_mre_mask = hv_ds_mre_mask.apply.opts(alpha=slider.param.value)

    layout = (((hv_ds_mre_image * hv_ds_mre_mask).grid('mre_type')) +
              (hv_ds_mri_image * hv_ds_mri_mask).layout('sequence').cols(3)
              ).cols(1)
    pn_layout = pn.pane.HoloViews(layout)
    wb = pn_layout.widget_box
    wb.append(slider)
    wb.append(cslider)
    wb.append(cslider2)

    # return pn.Column(slider, cslider2, layout, cslider)
    return pn.Column(wb, pn_layout)
    # return hv_ds_mri_image
    # return hv_ds_mre_image


def roc_curves(df, true='mre', pred='baseline', threshold=4, label=None, title=None, ax=None,
               plot=True):
    if label is None:
        label = pred
    if title is None:
        title = f'Threshold={threshold} kPa'
    threshold = threshold*1000
    pred_probs = torch.sigmoid(
        torch.Tensor((df[f'{pred}'].values)-threshold)/df[f'{pred}'].std()).numpy()
    true_labels = (df[f'{true}'] >= threshold).values.astype(int)
    df_labels = pd.DataFrame({'true_labels': true_labels, 'pred_probs': pred_probs})
    fpr, tpr, _ = roc_curve(df_labels.true_labels, df_labels.pred_probs)
    roc_auc = auc(fpr, tpr)
    # print('true_auc:', roc_auc)
    auc_std = []
    for i in range(100):
        df_labels_boot = df_labels.sample(n=len(df_labels), replace=True)
        fpr_boot, tpr_boot, _ = roc_curve(df_labels_boot.true_labels, df_labels_boot.pred_probs)
        roc_auc_boot = auc(fpr_boot, tpr_boot)
        auc_std.append(roc_auc_boot)
        # print('boot_auc:', roc_auc_boot)
    auc_std = np.std(auc_std)
    lw = 2
    if plot:
        if ax is None:
            plt.plot(fpr, tpr, lw=lw, label=f'{label} (AUROC = {roc_auc:0.2f})')
            plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', size=20)
            plt.ylabel('True Positive Rate', size=20)
            plt.title(title, size=22)
            plt.legend(loc="lower right", fontsize=15)
        else:
            ax.plot(fpr, tpr, lw=lw, label=f'{label} (AUROC = {roc_auc:0.2f})')
            # ax.plot(fpr, tpr, lw=lw, label=f'{label}')
            ax.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', size=20)
            ax.set_ylabel('True Positive Rate', size=20)
            ax.set_title(title, size=22)
            ax.legend(loc="lower right", fontsize=15)

    tp = df.query(f'{true}>{threshold} and {pred}>{threshold}').count()[0]
    fn = df.query(f'{true}>{threshold} and {pred}<{threshold}').count()[0]
    sens = tp/(tp+fn)
    # print('tp', tp)
    # print('fn', fn)
    # print('sensitivity', tp/(tp+fn))

    # Specificity: TN/(TN+FP)
    tn = df.query(f'{true}<{threshold} and {pred}<{threshold}').count()[0]
    fp = df.query(f'{true}<{threshold} and {pred}>{threshold}').count()[0]
    spec = tn/(tn+fp)
    # print('tn', tn)
    # print('fp', fp)
    # print('specificity', spec)

    acc = (tn+tp)/(tn+fp+tp+fn)
    sens_std = []
    spec_std = []
    acc_std = []

    # print('accuracy', acc)
    for i in range(100):
        df_boot = df.sample(n=len(df), replace=True)
        tp_boot = df_boot.query(f'{true}>{threshold} and {pred}>{threshold}').count()[0]
        fn_boot = df_boot.query(f'{true}>{threshold} and {pred}<{threshold}').count()[0]
        sens_boot = tp_boot/(tp_boot+fn_boot)
        # print('tp', tp)
        # print('fn', fn)
        # print('sensitivity', tp/(tp+fn))

        # Specificity: TN/(TN+FP)
        tn_boot = df_boot.query(f'{true}<{threshold} and {pred}<{threshold}').count()[0]
        fp_boot = df_boot.query(f'{true}<{threshold} and {pred}>{threshold}').count()[0]
        spec_boot = tn_boot/(tn_boot+fp_boot)
        # print('tn', tn)
        # print('fp', fp)
        # print('specificity', spec)

        acc_boot = (tn_boot+tp_boot)/(tn_boot+fp_boot+tp_boot+fn_boot)
        # print('acc boot', acc_boot)

        sens_std.append(sens_boot)
        spec_std.append(spec_boot)
        acc_std.append(acc_boot)

    sens_std = np.std(sens_std)
    spec_std = np.std(spec_std)
    acc_std = np.std(acc_std)

    return sens, sens_std, spec, spec_std, acc, acc_std, roc_auc, auc_std


def example_images(ds, subj='0219', z=18):
    '''Make a 2x3 grid of images for the important concepts of the analysis'''
    from matplotlib.colors import ListedColormap
    mask_map = ListedColormap(['black', 'powderblue'])

    ds_other = ds.sel(subject=subj, z=z)
    ds_mri = ds.sel(subject=subj, z=20)
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 10))
    fig.patch.set_alpha(1)
    mri = ds_mri.sel(sequence='t1_pre_water')['image_mri']
    ax[0][0].imshow(mri.T, cmap='gray', vmin=0, vmax=700)
    ax[0][0].axis('off')
    ax[0][0].set_title('MRI (Input Image)', size=18)
    ax[0][0].set_ylim(235, 15)
    ax[0][0].set_xlim(15, 235)
    ax[0][0].annotate("a.", xy=(-0.02, 1.04), xycoords="axes fraction", fontsize=16)

    mre = ds_other.sel(mre_type='mre_mask')['image_mre']/1000
    mre_cb = ax[0][1].imshow(mre.T, cmap='magma', vmin=0, vmax=7.800)
    ax[0][1].axis('off')
    ax[0][1].set_title('MRE (Hatched)', size=18)
    ax[0][1].set_ylim(235, 15)
    ax[0][1].set_xlim(15, 235)
    ax[0][1].annotate("b.", xy=(-0.02, 1.04), xycoords="axes fraction", fontsize=16)

    mre_pred = ds_other.sel(mre_type='pred')['image_mre']/1000
    ax[0][2].imshow(mre_pred.T, cmap='magma', vmin=0, vmax=7.800)
    ax[0][2].axis('off')
    ax[0][2].set_title('MRE (Predicted)', size=18)
    ax[0][2].set_ylim(235, 15)
    ax[0][2].set_xlim(15, 235)
    ax[0][2].annotate("c.", xy=(-0.02, 1.04), xycoords="axes fraction", fontsize=16)

    liver_mask = ds_other.sel(mask_type='liver')['mask_mri']
    liver_mask = morphology.remove_small_objects(liver_mask.values.astype(bool), min_size=1000)
    ax[1][0].imshow(liver_mask.T, cmap=mask_map)
    ax[1][0].axis('off')
    ax[1][0].set_title('Liver Segmentation', size=18)
    ax[1][0].set_ylim(235, 15)
    ax[1][0].set_xlim(15, 235)
    ax[1][0].annotate("d.", xy=(-0.02, 1.04), xycoords="axes fraction", fontsize=16)

    mre_mask = ds_other.sel(mask_type='mre')['mask_mre']
    ax[1][1].imshow(mre_mask.T, cmap=mask_map)
    ax[1][1].axis('off')
    ax[1][1].set_title('MRE Segmentation', size=18)
    ax[1][1].set_ylim(235, 15)
    ax[1][1].set_xlim(15, 235)
    ax[1][1].annotate("e.", xy=(-0.02, 1.04), xycoords="axes fraction", fontsize=16)

    combo_mask = ds_other.sel(mask_type='combo')['mask_mre']
    ax[1][2].imshow(combo_mask.T, cmap=mask_map)
    ax[1][2].axis('off')
    ax[1][2].set_title('ROI', size=18)
    ax[1][2].set_ylim(235, 15)
    ax[1][2].set_xlim(15, 235)
    ax[1][2].annotate("f.", xy=(-0.02, 1.04), xycoords="axes fraction", fontsize=16)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.55, 0.03, 0.3])
    cbar_ax.set_title('Stiffness\n(kPa)', size=16)
    cbar_ax.tick_params(labelsize=16)
    fig.colorbar(mre_cb, cax=cbar_ax)


def example_images_dice(ds, subj='0219', z=18):
    '''Make a 2x3 grid of images for the important concepts of the analysis'''
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    dice_map_true = ListedColormap(['deepskyblue'])
    dice_map_pred = ListedColormap(['C1'])
    dice_map_inter = ListedColormap(['maroon'])
    mask_map = ListedColormap(['white', 'grey'])
    # dice_map.set_bad(alpha=1)

    ds_other = ds.sel(subject=subj, z=z)
    ds_mri = ds.sel(subject=subj, z=20)
    fig, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(14, 10))
    fig.patch.set_alpha(1)
    mri = ds_mri.sel(sequence='t1_pre_water')['image_mri']
    ax[0][0].imshow(mri.T, cmap='gray', vmin=0, vmax=700)
    ax[0][0].axis('off')
    ax[0][0].set_title('MRI (Input Image)', size=18)
    ax[0][0].set_ylim(235, 15)
    ax[0][0].set_xlim(15, 235)

    mre = ds_other.sel(mre_type='mre_mask')['image_mre']/1000
    mre_cb = ax[0][1].imshow(mre.T, cmap='magma', vmin=0, vmax=7.800)
    ax[0][1].axis('off')
    ax[0][1].set_title('MRE (Hatched)', size=18)
    ax[0][1].set_ylim(235, 15)
    ax[0][1].set_xlim(15, 235)

    mre_pred = ds_other.sel(mre_type='pred')['image_mre']/1000
    ax[0][2].imshow(mre_pred.T, cmap='magma', vmin=0, vmax=7.800)
    ax[0][2].axis('off')
    ax[0][2].set_title('MRE (Predicted)', size=18)
    ax[0][2].set_ylim(235, 15)
    ax[0][2].set_xlim(15, 235)

    combo_mask = ds_other.sel(mask_type='combo')['mask_mre']
    ax[1][0].imshow(combo_mask.T, cmap=mask_map, interpolation='none', alpha=1)
    ax[1][0].axis('off')
    ax[1][0].set_title('ROI (zoomed)', size=18)
    ax[1][0].set_ylim(170, 45)
    ax[1][0].set_xlim(30, 170)

    true_dice_f1 = np.ma.masked_where(mre*combo_mask >= 2.88, np.ones_like(mre))
    true_dice_f1 = np.ma.masked_array(np.ones_like(mre), ~true_dice_f1.mask)
    pred_dice_f1 = np.ma.masked_where(mre_pred*combo_mask >= 2.88, np.ones_like(mre_pred))
    pred_dice_f1 = np.ma.masked_array(np.ones_like(mre), ~pred_dice_f1.mask)
    ax[1][1].imshow(pred_dice_f1.T, cmap=dice_map_pred, interpolation='none', label='Predicted')
    ax[1][1].imshow(true_dice_f1.T, cmap=dice_map_true, interpolation='none', label='True')
    ax[1][1].imshow(true_dice_f1.T*pred_dice_f1.T, cmap=dice_map_inter, interpolation='none',
                    label='Overlap')
    ax[1][1].axis('off')
    ax[1][1].set_title('2.88 kPa or Greater', size=18)
    ax[1][1].set_ylim(170, 45)
    ax[1][1].set_xlim(30, 170)

    true_dice_f4 = np.ma.masked_where(mre*combo_mask >= 4.09, np.ones_like(mre))
    true_dice_f4 = np.ma.masked_array(np.ones_like(mre), ~true_dice_f4.mask)
    pred_dice_f4 = np.ma.masked_where(mre_pred*combo_mask >= 4.09, np.ones_like(mre_pred))
    pred_dice_f4 = np.ma.masked_array(np.ones_like(mre), ~pred_dice_f4.mask)
    ax[1][2].imshow(pred_dice_f4.T, cmap=dice_map_pred, interpolation='none', label='Predicted')
    ax[1][2].imshow(true_dice_f4.T, cmap=dice_map_true, interpolation='none', label='True')
    ax[1][2].imshow(true_dice_f4.T*pred_dice_f4.T, cmap=dice_map_inter, interpolation='none',
                    label='Overlap')
    ax[1][2].axis('off')
    ax[1][2].set_title('4.09 kPa or Greater', size=18)
    ax[1][2].set_ylim(170, 45)
    ax[1][2].set_xlim(30, 170)
    pred_patch = mpatches.Patch(color='C1', label='Predicted')
    true_patch = mpatches.Patch(color='deepskyblue', label='True')
    overlap_patch = mpatches.Patch(color='maroon', label='Overlap')
    ax[1][2].legend(handles=[true_patch, pred_patch, overlap_patch], loc='lower right')
    ax[1][1].legend(handles=[true_patch, pred_patch, overlap_patch], loc='lower right')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.55, 0.03, 0.3])
    cbar_ax.set_title('Stiffness\n(kPa)', size=16)
    cbar_ax.tick_params(labelsize=16)
    fig.colorbar(mre_cb, cax=cbar_ax)


def model_feature_extractor(ds, model_path=None, subj='0219'):
    '''Load in a model, then get the next-to-last layer.'''

    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                       fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12},
                       plot_size=300),
        opts.Image(cmap='viridis', width=550, height=550, tools=['hover'], xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='20pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.Overlay(show_legend=True))

    model = DeepLabFeatures(in_channels=3, out_channels=1, output_stride=8, do_ord=False, norm='bn',
                            do_clinical=True)

    if model_path is None:
        model_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data',
                          'trained_models', 'GROUP16',
                          'model_2020-04-15_10-55-56_n0.pkl')

    model_dict = torch.load(model_path, map_location='cuda:0')
    model_dict = OrderedDict([(key[7:], val) for key, val in model_dict.items()])
    model.load_state_dict(model_dict, strict=False)
    model.to('cuda:0')
    model.eval()

    clinical = ['age', 'gender', 'height', 'weight', 'bmi', 'htn', 'hld', 'dm', 'ast', 'alt', 'alk',
                'tbili', 'albumin', 'plt']
    mean = ds[clinical].mean()
    mean = np.asarray([mean[val].values for val in clinical])
    std = ds[clinical].std()
    std = np.asarray([std[val].values for val in clinical])

    torch_data = MRETorchDataset(ds.sel(subject=[subj]), set_type='test',
                                 inputs=['t1_pre_water', 't1_pre_fat', 't1_pos_70_water'],
                                 do_clinical=True, norm_clin_vals=[mean, std], erode_mask=3)
    dataloader = DataLoader(torch_data, batch_size=1, shuffle=False, num_workers=1)
    data = next(iter(dataloader))
    inputs = data[0].to('cuda:0')
    clinical = data[4].to('cuda:0')
    outputs = model(inputs, clinical).cpu().detach().numpy()[0]
    xa = xr.DataArray(outputs, dims=['features', 'z', 'y', 'x'])
    xa.name = 'deeplab'
    hv_xa = hv.Dataset(xa)
    hv_image = hv_xa.to(hv.Image, kdims=['x', 'y'], vims='deeplab',
                        dynamic=True).opts(invert_yaxis=True)
    # hv_image = hv_image.redim.range((-1, 1))
    return hv_image


def radiology_cor_plots(ds, df=None, do_aug=True, do_cor=True,
                        pred='pred', save_name='test', plot=True):
    import seaborn as sns
    sns.set()
    sns.set_palette(sns.color_palette('colorblind'))

    true_pixel = []
    pred_pixel = []
    true_subj = []
    pred_subj = []
    # if do_cor:
    #     slope = np.mean(ds['val_slope'].values)
    #     intercept = np.mean(ds['val_intercept'].values)
    #     print(slope, intercept)
    for subj in ds.subject:
        mask = ds.sel(subject=subj, mask_type='combo')['mask_mri'].values.copy()
        # print(mask.shape)
        # print(mask.mean())
        if do_aug:
            for i in range(mask.shape[2]):
                if mask[:, :, i].mean() > 0:
                    mask[:, :, i] = ndi.binary_erosion(
                        mask[:, :, i], iterations=2).astype(mask.dtype)
        mask = np.where(mask > 0, mask, np.nan)
        true_mre_region = (ds.sel(subject=subj, mre_type='mre')['image_mre'].values * mask)
        true_mre_region = true_mre_region.flatten()
        true_mre_region = true_mre_region[~np.isnan(true_mre_region)]
        pred_mre_region = (ds.sel(subject=subj, mre_type=pred)['image_mre'].values * mask)
        pred_mre_region = pred_mre_region.flatten()
        pred_mre_region = pred_mre_region[~np.isnan(pred_mre_region)]
        if do_cor:
            slope = np.mean(ds.sel(subject=subj, mre_type=pred)['val_slope'].values)
            intercept = np.mean(ds.sel(subject=subj, mre_type=pred)['val_intercept'].values)
            # print(slope, intercept)
            pred_mre_region = (pred_mre_region-intercept)/slope
            pred_mre_region = np.where(pred_mre_region > 0, pred_mre_region, 0)
        true_pixel.append(true_mre_region/1000)
        pred_pixel.append(pred_mre_region/1000)
        true_subj.append(np.nanmean(true_mre_region)/1000)
        pred_subj.append(np.nan_to_num(np.nanmean(pred_mre_region/1000)))

    true_pixel = np.concatenate(true_pixel)
    pred_pixel = np.concatenate(pred_pixel)
    df_subj = pd.DataFrame({'true': true_subj, 'predict': pred_subj, 'subject': ds.subject.values})
    df_subj['fibrosis'] = np.where(df_subj.true > 3.77, 'Severe Fibrosis', 'Mild Fibrosis')

    # Subj Plot
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(df_subj['predict'], params, x=df_subj['true'])
    r2_subj_std = []
    for i in range(100):
        df_subj_boot = df_subj.sample(n=len(df_subj), replace=True)
        model_boot = LinearModel()
        params_boot = model.make_params(slope=1, intercept=0)
        result_boot = model_boot.fit(df_subj_boot['predict'], params_boot, x=df_subj_boot['true'])
        r2_subj_boot = 1 - result_boot.residual.var() / np.var(df_subj_boot['predict'])
        r2_subj_std.append(r2_subj_boot)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ymax = df_subj['predict'].max()+1
        xmax = df_subj['true'].max()+1
        rect1 = plt.Rectangle((0, 0), 3.77, 3.77, color='grey', alpha=0.3,
                              label='Correct Classification\n3.77 kPa Threshold')
        rect2 = plt.Rectangle((3.77, 3.77), xmax-3.77, ymax-3.77, color='grey', alpha=0.3)
        ax[0].add_patch(rect1)
        ax[0].add_patch(rect2)
        df_subj.query('true>3.77').plot(x='true', y='predict', kind='scatter',
                                        xlim=(0, xmax), ylim=(0, ymax), color='C1', ax=ax[0],
                                        label='High Stiffness')
        df_subj.query('true<3.77').plot(x='true', y='predict', kind='scatter',
                                        xlim=(0, xmax), ylim=(0, ymax), color='C0', ax=ax[0],
                                        label='Low Stiffness')
        ax[0].plot(df_subj['true'], result.best_fit, label='Best Fit', linewidth=2, c='k')
        ax[0].set_title('Predicted vs True Average Stiffness', size=22)
        # plt.ylim(-1, df_subj['predict'].max[0]())
        # plt.xlim(-1, df_subj['predict'].max[0]())
        ax[0].set_xlabel('True Stiffness (kPa)', size=20)
        ax[0].set_ylabel('Predicted Stiffness (kPa)', size=20)
        lgnd = ax[0].legend(fontsize=15, frameon=False)
        for handle in lgnd.legendHandles:
            try:
                handle.set_sizes([60])
            except AttributeError:
                pass
        ax[0].tick_params(labelsize=16)
        ax[0].annotate('True Positive', (8, 8), size=15, weight='bold')
        ax[0].annotate('True\nNegative', (0.3, 2), size=15, weight='bold')
        ax[0].annotate('False Negative', (8, 3), size=15, weight='bold')
        ax[0].annotate('False Postive', (1, 7), size=15, weight='bold')
        # plt.savefig(f'../plots/subj_results_{save_name}.pdf', bbox_inches='tight')

        print(result.fit_report())
    r2_subj = 1 - result.residual.var() / np.var(df_subj['predict'])
    print('R2 (subj):', r2_subj)

    # Pixel Plot
    model = LinearModel()
    params = model.make_params(slope=1, intercept=0)
    result = model.fit(pred_pixel, params, x=true_pixel)
    r2_pixel_std = []
    for i in range(100):
        idx = np.random.choice(np.arange(len(true_pixel)), 1000, replace=True)
        pred_pixel_boot = pred_pixel[idx]
        true_pixel_boot = true_pixel[idx]
        model_boot = LinearModel()
        params_boot = model.make_params(slope=1, intercept=0)
        result_boot = model_boot.fit(pred_pixel_boot, params_boot, x=true_pixel_boot)
        r2_pixel_boot = 1 - result_boot.residual.var() / np.var(pred_pixel_boot)
        r2_pixel_std.append(r2_pixel_boot)
    if plot:
        thresholds = ['2.88', '3.54', '3.77', '4.09']

        for i, thresh in enumerate(thresholds):
            _ = roc_curves(df, pred='pred', label=thresh, threshold=float(thresh), ax=ax[1])
            ax[1].set_title('ROC Curves', size=22)
            leg = ax[1].get_legend()
            leg.set_title('Stiffness Threshold (kPa)', prop={'size': 15})
        # hb = ax[1].hexbin(true_pixel, pred_pixel, bins=[0,10,100,1000,10000], cmap='Blues',
        #                   extent=(0, 15, 0, 15), gridsize=20)
        # hb = ax[1].hist2d(true_pixel, pred_pixel,
        #                   bins=[0, 2.88, 3.54, 3.77, 4.09, 10], cmap='Blues')
        # ax[1].hist(true_pixel-pred_pixel, bins=50)
        # ax[1].plot(true_pixel, result.best_fit, label='Best Fit', linewidth=2, c='k')
        # cb = fig.colorbar(hb, ax=ax[1])
        # cb.set_label('log(# of Pixels)', fontsize=20)
        # cb.ax.tick_params(labelsize=16)

        # ax[1].set_title('Pixel-wise Difference', size=22)
        # ax[1].set_ylim(0, 15)
        # ax[1].set_xlim(-7, 7)
        # ax[1].set_xlabel('True - Predicted Stiffness (kPa)', size=20)
        # ax[1].set_ylabel('Pixel Count', size=20)
        # ax[1].tick_params(labelsize=16)
        # ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # ax[1].yaxis.offsetText.set_fontsize(14)
        # plt.savefig(f'../plots/pixel_results_{save_name}.pdf', bbox_inches='tight')

        print(result.fit_report())
    r2_pixel = 1 - result.residual.var() / np.var(pred_pixel)
    print('R2:', r2_pixel)

    if plot:
        # Sensitivity S: TP/(TP+FN)
        tp = df_subj.query('true>3.77 and predict>3.77').count()[0]
        fn = df_subj.query('true>3.77 and predict<3.77').count()[0]
        print('tp', tp)
        print('fn', fn)
        print('sensitivity', tp/(tp+fn))

        # Specificity: TN/(TN+FP)
        tn = df_subj.query('true<3.77 and predict<3.77').count()[0]
        fp = df_subj.query('true<3.77 and predict>3.77').count()[0]
        print('tn', tn)
        print('fp', fp)
        print('specificity', tn/(tn+fp))

        print('accuracy', (tn+tp)/(tn+fp+tp+fn))
        print(df_subj.true.mean())
        print(df_subj.true.std())
    r2_subj_std = np.std(r2_subj_std)
    r2_pixel_std = np.std(r2_pixel_std)

    return r2_subj, r2_subj_std, r2_pixel, r2_pixel_std
