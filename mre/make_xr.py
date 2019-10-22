import time
import copy
from pathlib import Path
import warnings
import argparse
from collections import defaultdict
import pickle as pkl
import numpy as np
from itertools import chain
import xarray as xr
from mre.mre_datasets import MREtoXr


def make_xr(data_dir: str, subj: str, sequences: list, verbose: str = True, **kwargs) -> None:
    '''Function to start training liver segmentation.

    This function is intended to be imported and called in interactive sessions, from the command
    line, or by a slurm job submission script.  This function should be able to tweak any part of
    the model (or models) via a standardized config in order to easily document the impact of
    parameter settings.

    Args:
        data_dir of data (str): Full path to location of data.
        patient (str): Name of patient (unique ID).
        sequences (str): Input sequences wanted.
        verbose (str): Print or suppress cout statements.

    Returns:
        None
    '''

    # cfg = process_kwargs(kwargs)
    data_dir = Path(data_dir[1:-1])
    xr_maker = MREtoXr(data_dir, None, subj, output_name=subj)
    _  = xr_maker.load_xr()


def default_cfg():
    cfg = {'sequences': ['t1_pre_water', 't1_pre_in', 't1_pre_out', 't2']}
    return cfg


def str2bool(val):
    if type(val) is not str:
        return val
    elif val.lower() in ("yes", "true", "t"):
        return True
    elif val.lower() in ("no", "false", "f"):
        return False
    else:
        return val


def process_kwargs(kwargs):
    cfg = default_cfg()
    for key in kwargs:
        val = str2bool(kwargs[key])
        cfg[key] = val
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument(
        '--data_dir', type=str, help='Path to input data.',
        default='/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/MR/')
    parser.add_argument('--subj', type=str, help='Name of patient.',
                        default='0006')
    parser.add_argument('--sequences', type=str, nargs='+', help='Input sequences.',
                        default="t1_pre_water t1_pre_in t1_pre_out t2")
    parser.add_argument('--verbose', type=bool, help='Verbose printouts.',
                        default=True)
    # cfg = default_cfg()
    # for key in cfg:
    #     val = str2bool(cfg[key])
    #     if type(val) is bool:
    #         parser.add_argument(f'--{key}', action='store', type=str2bool,
    #                             default=val)
    #     else:
    #         parser.add_argument(f'--{key}', action='store', type=type(val),
    #                             default=val)

    args = parser.parse_args()
    make_xr(**vars(args))
