from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import SimpleITK as sitk
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from mre.preprocessing import MRIImage
from lmfit.models import LinearModel

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
            inputs=(0, 9)).opts(cmap='viridis', title=f'Input')

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
    hv_ds_mri = hv.Dataset(xr_ds[['image_mri', 'mask_mri']])
    hv_ds_mre = hv.Dataset(xr_ds[['image_mre', 'mask_mre']])
    if prediction:
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre', 'mre_mask', 'mre_pred'])
        hv_ds_mre_2 = hv_ds_mre.select(mre_type=['mre_raw', 'mre_wave'])
    elif not torch:
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre', 'mre_mask'])
        hv_ds_mre_2 = hv_ds_mre.select(mre_type=['mre_raw', 'mre_wave'])
    else:
        hv_ds_mre_1 = hv_ds_mre.select(mre_type=['mre'])
    print(hv_ds_mri)
    print(hv_ds_mre)

    hv_ds_mri_image = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='image_mri', dynamic=True)
    hv_ds_mri_mask = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='mask_mri',
                                  dynamic=True).opts(tools=[])

    hv_ds_mre_image_1 = hv_ds_mre_1.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                       dynamic=True).opts(cmap='viridis')
    hv_ds_mre_mask_1 = hv_ds_mre_1.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                      dynamic=True).opts(tools=[])
    if not torch:
        hv_ds_mre_image_2 = hv_ds_mre_2.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                           dynamic=True).opts(cmap='viridis')
        hv_ds_mre_mask_2 = hv_ds_mre_2.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                          dynamic=True).opts(tools=[])

    slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='mask transparency')
    if torch:
        cslider = pn.widgets.RangeSlider(start=-2, end=2, value=(-2, 2), name='contrast')
        cslider2 = pn.widgets.RangeSlider(start=0, end=200, value=(0, 100), name='mre contrast')
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

    if not torch:
        redim_image_mre_2 = {'image_mre': (-200, 200)}
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

    # return pn.Column(slider, cslider2, layout, cslider)
    return pn.Column(wb, pn_layout)
    # return hv_ds_mri_image
    # return hv_ds_mre_image


def miccai_plots(ds, do_cor=True, save_name='test'):
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
