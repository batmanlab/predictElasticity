import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


def grid_plots(ds, rows, cols, title=None, xlabel=None, ylabel=None):
    '''helper function to automate the creation of grid plots'''
    n_rows = len(ds[rows])
    n_cols = len(ds[cols])
    fig, axs = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(3.75*n_cols, 3.75*n_rows),
                            sharex=True, sharey=True)
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
