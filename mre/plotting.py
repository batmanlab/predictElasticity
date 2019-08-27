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
    targets = targets.redim.range(target=(0, 70)).opts(title='Target')*masks

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
                                                  prediction=(0, 50)).opts(title='Prediction')*masks

        layout = hv.Layout(input_list + [targets] + [predictions]).cols(3)
    else:
        layout = hv.Layout(input_list + [targets]).cols(3)
    return pn.Column(slider, layout)


def torch_dl_vis(inputs, targets, masks):
    print('lol')
    # titles = ['T1Pre', 'T1Pos', 'T2SS', 'T2FR', 'elast', 'mask']
    titles = ['T1Pre', 'T1Pos', 'T2SS', 'elast', 'mask']
    n_input = inputs.shape[0]
    n_seq = inputs.shape[1]+targets.shape[1]+masks.shape[1]
    fig, axes = plt.subplots(n_input, n_seq, tight_layout=True,
                             figsize=(2.5*n_seq, 2.5*n_input), sharex=True, sharey=True)
    for i in range(n_input):
        for seq in range(n_seq):
            if seq < inputs.shape[1]:
                # set_trace()
                axes[i][seq].imshow(np.asarray(inputs[i, seq, :, :]))
            elif seq < inputs.shape[1]+targets.shape[1]:
                axes[i][seq].imshow(np.asarray(targets[i, seq-inputs.shape[1], :, :]))
            else:
                axes[i][seq].imshow(np.asarray(masks[i,
                                                     seq-inputs.shape[1]-targets.shape[1], :, :]))

            axes[i][seq].axis('off')
            if i == 0:
                axes[i][seq].set_title(titles[seq])
        if seq == 0:
            axes[i][seq].set_xlabel(f'subj = {i}')
    fig.lines.append(plt.vlines(0.8, 0, 1, transform=fig.transFigure))
    plt.show()


def hv_alpha_plots(ds, seq_list=None):
    if seq_list is None:
        seq_list = [seq for seq in ds.sequence.values if 'extra' not in seq]

    if len(seq_list) == 2:
        color_list = ['viridis', 'Reds']
    else:
        color_list = ['Blues', 'Greens', 'Reds', 'Purples', 'Greys', 'Oranges']

    hv_dict = {}
    dmap_dict = {}
    slider_dict = {}

    for i, seq in enumerate(seq_list):
        if (len(seq_list) == 2 and i == 0):
            hv_dict[seq] = hv.Dataset(ds['image'].sel(sequence=seq).copy())
            dmap_dict[seq] = hv_dict[seq].to(hv.Image, groupby=['z', 'subject'], dynamic=True)
            dmap_dict[seq] = dmap_dict[seq].opts(style=dict(cmap=color_list[i]),
                                                 plot=dict(width=500, height=500, tools=['hover']))
        elif (len(seq_list) == 2 and i == 1):
            mask = ds['image'].sel(sequence=seq)
            mask = mask.where(mask < 1, 1)
            mask = mask.where(mask > 0, np.nan)
            hv_dict[seq] = hv.Dataset(mask)
            dmap_dict[seq] = hv_dict[seq].to(hv.Image, groupby=['z', 'subject'], dynamic=True)
            dmap_dict[seq] = dmap_dict[seq].opts(style=dict(cmap=color_list[i]),
                                                 plot=dict(width=500, height=500, tools=['hover']))
            slider_dict[seq] = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name=seq)
            dmap_dict[seq] = dmap_dict[seq].apply.opts(alpha=slider_dict[seq].param.value)
            dmap_dict[seq] = dmap_dict[seq].redim.range(image=(0, 500))
        else:
            hv_dict[seq] = hv.Dataset(ds['image'].sel(sequence=seq).copy())
            dmap_dict[seq] = hv_dict[seq].to(hv.Image, groupby=['z', 'subject'], dynamic=True)
            dmap_dict[seq] = dmap_dict[seq].opts(style=dict(cmap=color_list[i]),
                                                 plot=dict(width=500, height=500, tools=['hover']))
            slider_dict[seq] = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name=seq)
            dmap_dict[seq] = dmap_dict[seq].apply.opts(alpha=slider_dict[seq].param.value)
            dmap_dict[seq] = dmap_dict[seq].redim.range(image=(0, 500))
    overlay = reduce((lambda x, y: x * y), dmap_dict.values())

    return pn.Column(*slider_dict.values(), overlay)
    # return slider_dict, hv_dict, overlay, dmap_dict


def hv_comp_plots(ds, seq_list=None, mask=None, mask_trim=0):
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
        opts.Image(cmap='viridis', width=350, height=350, tools=['hover'], xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='8pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.Overlay(show_legend=False))
    if seq_list is None:
        seq_list = [seq for seq in ds.sequence.values if 'extra' not in seq]

    hv_dict = {}
    dmap_dict = {}

    for i, seq in enumerate(seq_list):
        hv_dict[seq] = hv.Dataset(ds['image'].sel(sequence=seq).copy())
        dmap_dict[seq] = hv_dict[seq].to(hv.Image, groupby=['z', 'subject'], dynamic=True)
        dmap_dict[seq] = dmap_dict[seq].opts(style=dict(cmap='viridis'),
                                             plot=dict(width=500, height=500, tools=['hover']))
        if seq in ['elast', 'elastMsk', 'mre_pred']:
            dmap_dict[seq] = dmap_dict[seq].redim(image='mre')
            dmap_dict[seq] = dmap_dict[seq].redim.range(mre=(0, 10000))
        else:
            dmap_dict[seq] = dmap_dict[seq].redim(image=f'image{i}')
    if mask is not None:
        hv_mask = hv.Dataset(ds['image'].sel(sequence=mask).copy())
        hv_mask.data = hv_mask.data.where(hv_mask.data > mask_trim)
        dmap_mask = hv_mask.to(hv.Image, groupby=['z', 'subject'], dynamic=True)
        dmap_mask = dmap_mask.opts(style=dict(cmap='reds'),
                                   plot=dict(width=500, height=500, tools=[]))
        slider = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name='mask')
        dmap_mask = dmap_mask.apply.opts(alpha=slider.param.value)
        dmap_mask = dmap_mask.redim(image='mask')
        dmap_mask = dmap_mask.redim.range(mask=(0, 2))
        overlay = reduce((lambda x, y: x*dmap_mask + y*dmap_mask), dmap_dict.values())

        return pn.Column(slider, hv.Layout(overlay).cols(3))
    else:
        overlay = reduce((lambda x, y: x + y), dmap_dict.values())
        return pn.Column(hv.Layout(overlay).cols(3))


def patient_series_viewer(path, patient, img_type='DICOM', info=''):
    '''Similar to pybreast viewer, but must natively handle a mix of 2d, 3d, scalar, and vector'''

    imopts = {'tools': ['hover'], 'width': 500, 'height': 500, 'cmap': 'viridis'}
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

    else:
        raise KeyError(f'img_type must be one of ["DICOM", "NIFTI"], got {img_type}')

    hv_images = []
    for img_files in img_folders:
        print(img_files)
        hvds_list = []
        if 'DICOM' in img_type:
            dicom_names = reader.GetGDCMSeriesFileNames(str(img_files))
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()  # Get DICOM Info
            reader.LoadPrivateTagsOn()  # Get DICOM Info
            image = reader.Execute()
            desc = reader.GetMetaData(0, '0008|103e').strip().encode('utf-8', 'ignore').decode()
            pid = reader.GetMetaData(0, '0010|0010').strip()
            desc = ' '.join([img_files.stem, desc, pid])
        elif img_type == 'NIFTI':
            reader.SetFileName(str(img_files))
            desc = ' '.join(img_files.parts[-2:])
            image = reader.Execute()

        npimg = sitk.GetArrayFromImage(image)
        print(npimg.shape)
        if npimg.shape[0] == 1:
            hv_images.append(hv.Image(npimg[0, :], label=desc).opts(**imopts))
        elif npimg.shape[-1] > 3:
            hvds_list.append(hv.Dataset(
                (np.arange(npimg.shape[2]), np.arange(npimg.shape[1]), np.arange(npimg.shape[0]),
                 npimg), [f'x{desc}', f'y{desc}', f'z{desc}'],
                f'MRI{desc}'))
            print(hvds_list[-1])
            hv_images.append(hvds_list[-1].to(hv.Image, [f'x{desc}', f'y{desc}'],
                                              groupby=[f'z{desc}'],
                                              dynamic=True, label=desc).opts(**imopts,
                                                                             invert_yaxis=True))
        else:
            hv_images.append(hv.Image(npimg[0, :], label=desc).opts(**imopts))
    return hv.Layout(hv_images).opts(shared_axes=False, merge_tools=False, normalize=False,
                                     title=' '.join([patient, info])).cols(2)


def chaos_viewer(path, patient):
    '''Specifically for viewing CHAOS images with masks, assumes NIFTI'''

    clipping = {'min': 'transparent', 'max': 'red', 'NaN': 'transparent'}
    imopts = {'tools': ['hover'], 'width': 500, 'height': 500, 'cmap': 'viridis'}
    full_path = Path(path, patient)

    img_files = sorted(list(full_path.glob('*MR*.nii')))
    mask_files = sorted(list(full_path.glob('*mask*.nii')))

    hv_images = []
    sliders = []
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
        sliders.append(slider)

        hv_img = ds_img.to(hv.Image, [f'x{desc}', f'y{desc}'],
                           groupby=[f'z{desc}'],
                           dynamic=True, label=desc).opts(**imopts, invert_yaxis=True)

        hv_mask = ds_mask.to(hv.Image, [f'x{desc}', f'y{desc}'],
                             groupby=[f'z{desc}'],
                             dynamic=True, label=desc).opts(cmap='Reds', invert_yaxis=True,
                                                            clipping_colors=clipping)

        hv_mask = hv_mask.apply.opts(alpha=slider.param.value)
        vrange = {f'mask{desc}': (0, 10)}
        hv_mask = hv_mask.redim.range(**vrange)

        hv_images.append(hv_img * hv_mask)

    return pn.Column(*sliders, hv.Layout(hv_images).opts(shared_axes=False, normalize=False,
                                                         title=' '.join(patient)).cols(2))


def patient_reg_comparison(fixed, moving_init, moving_final, grid=None):
    '''Comparing 3 images at once for alpha blending.  Expects sitk input format.'''

    imopts = {'tools': ['hover'], 'width': 500, 'height': 500}
    hvds_fixed = hv.Dataset(MRIImage(fixed, 'fixed', 'fixed').da)
    hvds_moving_init = hv.Dataset(MRIImage(moving_init, 'moving_init', 'moving_init').da)
    hvds_moving_init = hvds_moving_init.redim(z='z1')
    print(hvds_moving_init)
    hvds_moving_final = hv.Dataset(MRIImage(moving_final, 'moving_final', 'moving_final').da)

    hv_fixed = hvds_fixed.to(hv.Image, kdims=['x', 'y'], groupby=['z'], dynamic=True)
    hv_fixed.opts(**imopts, cmap='viridis', title='fixed and moving_final')
    # hv_fixed.redim.range(fixed=(hvds_fixed.data.min().values, hvds_fixed.data.max().values))

    hv_moving_init = hvds_moving_init.to(hv.Image, kdims=['x', 'y'], groupby=['z1'], dynamic=True)
    hv_moving_init.opts(**imopts, cmap='Reds', title='moving_init')

    hv_moving_final = hvds_moving_final.to(hv.Image, kdims=['x', 'y'], groupby=['z'], dynamic=True)
    hv_moving_final.opts(**imopts, cmap='Reds', title='moving_final')
    if grid:
        hv_grid = hv.Image(sitk.GetArrayFromImage(grid), groupby=['z']).opts(**imopts,
                                                                             cmap='Greys_r')

    # Make an alpha slider
    slider1 = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name='moving_init')
    slider2 = pn.widgets.FloatSlider(start=0, end=1, value=0.0, name='moving_final')
    # Plot the slider and the overlayed images using the '*' operator
    if grid:
        return pn.Column(slider1, slider2,
                         rasterize(hv_fixed) *
                         rasterize(hv_moving_init.apply.opts(alpha=slider1.param.value)) *
                         rasterize(hv_moving_final.apply.opts(alpha=slider2.param.value)) +
                         rasterize(hv_grid))
    else:
        return pn.Column(slider2,
                         hv_fixed *
                         hv_moving_final.apply.opts(alpha=slider2.param.value) +
                         hv_moving_init)
