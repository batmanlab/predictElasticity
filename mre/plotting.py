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
