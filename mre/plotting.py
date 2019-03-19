import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import SimpleITK as sitk


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
            # if j == 0:
                # axs[i][j].set_ylabel(f'{row.values}', size=18)
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
