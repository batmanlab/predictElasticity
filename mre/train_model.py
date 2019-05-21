#! /usr/bin/env python3

import os
import argparse
import pickle as pkl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from mre.prediction_v2 import MREDataset


def train_model(data_path: str, data_file: str, output_path: str, model_version: str = 'tmp',
                verbose: str = True, **kwargs) -> None:
    '''Function to start training MRE model given a user-defined set of parameters.

    This function is intended to be imported and called in interactive sessions, from the command
    line, or by a slurm job submission script.  This function should be able to tweak any part of
    the model (or models) via a standardized config in order to easily document the impact of
    parameter settings.

    Args:
        data_path (str): Full path to location of data.
        data_file (str): Name of pickled data file.
        output_path (str): Full path to output directory.
        model_version (str): Name of model.
        verbose (str): Print or suppress cout statements.

    Returns:
        None
    '''
    # Load config and data
    cfg = process_kwargs(kwargs)
    ds = pkl.load(open(os.path.join(data_path, data_file), 'rb'))
    subj = cfg['subj']
    batch_size = cfg['batch_size']

    # Start filling dataloaders
    dataloaders = {}
    train_set = MREDataset(ds, set_type='train', transform=cfg['train_trans'],
                           clip=cfg['train_clip'], aug=cfg['train_aug'], test=subj)
    val_set = MREDataset(ds, set_type='val', transform=cfg['val_trans'],
                         clip=cfg['val_clip'], aug=cfg['val_aug'], test=subj)
    test_set = MREDataset(ds, set_type='test', transform=cfg['test_trans'],
                          clip=cfg['test_clip'], aug=cfg['test_aug'], test=subj)
    if cfg['train_sample'] == 'shuffle':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                          num_workers=0)
    elif cfg['train_sample'] == 'resample':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                          sampler=RandomSampler(
                                              train_set, replacement=True,
                                              num_samples=cfg['train_num_samples']),
                                          num_workers=0),
    if cfg['val_sample'] == 'shuffle':
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                        num_workers=0)
    elif cfg['val_sample'] == 'resample':
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                        sampler=RandomSampler(
                                            val_set, replacement=True,
                                            num_samples=cfg['val_num_samples']),
                                        num_workers=0),
    dataloaders['test'] = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                    num_workers=0)



def process_kwargs(kwargs):
    cfg = default_cfg()
    for key, val in kwargs:
        cfg[key] = val
    return cfg


def default_cfg():
    cfg = {'train_trans': True, 'train_clip': True, 'train_aug': True,
           'val_trans': True, 'val_clip': True, 'val_aug': False,
           'test_trans': True, 'test_clip': True, 'test_aug': False,
           'subj': 162, 'batch_size': 50,
           }
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                                            help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args.accumulate(args.integers))
