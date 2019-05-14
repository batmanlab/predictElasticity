import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import SimpleITK as sitk
import panel as pn
import holoviews as hv
from holoviews import opts
from functools import reduce
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
    ds_mask_liver = xr.DataArray(np.where(masks<1e-6, np.nan, 1),
                                 dims=['subject', 'sequence', 'y', 'x'],
                                 coords=[list(names), ['mask_liver'],
                                         range(0, -predictions.shape[2], -1),
                                         range(predictions.shape[3])
                                         ],
                                 name='mask_liver')
    ds_mask_conf = xr.DataArray(np.where(masks<1, np.nan, 1),
                                dims=['subject', 'sequence', 'y', 'x'],
                                coords=[list(names), ['mask_conf'],
                                        range(0, -predictions.shape[2], -1),
                                        range(predictions.shape[3])
                                        ],
                                name='mask_liver')
    ds_predictions = xr.DataArray(predictions,
                                  dims=['subject', 'sequence', 'y', 'x'],
                                  coords=[list(names), ['prediction'],
                                          range(0, -predictions.shape[2], -1),
                                          range(predictions.shape[3])
                                          ],
                                  name='prediction')
    for subj in names:
        print(np.nanmean(ds_mask_liver.sel(subject=subj).values*ds_targets.sel(subject=subj).values))
        print(np.nanmean(ds_mask_liver.sel(subject=subj).values*ds_predictions.sel(subject=subj).values))


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
