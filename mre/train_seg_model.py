#!/usr/bin/env python

from pathlib import Path
import warnings
import argparse
import pickle as pkl
import numpy as np
from itertools import chain
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from tensorboardX import SummaryWriter
from mre.segmentation import ChaosDataset
from mre.prediction_v2 import train_model
from mre import pytorch_arch
from robust_loss_pytorch import adaptive


def train_seg_model(data_path: str, data_file: str, output_path: str, model_version: str = 'tmp',
                    verbose: str = True, **kwargs) -> None:
    '''Function to start training liver segmentation.

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
    torch.manual_seed(100)
    cfg = process_kwargs(kwargs)
    if verbose:
        print(cfg)
    ds = xr.open_dataset(Path(data_path, data_file))
    if verbose:
        print(ds)
    subj = cfg['subj']
    batch_size = cfg['batch_size']
    loss_type = cfg['loss']

    # Start filling dataloaders
    dataloaders = {}
    train_set = ChaosDataset(ds, set_type='train', transform=cfg['train_trans'],
                             clip=cfg['train_clip'], aug=cfg['train_aug'],
                             sequence_mode=cfg['train_seq_mode'],
                             resize=cfg['resize'],
                             test=subj)
    val_set = ChaosDataset(ds, set_type='val', transform=cfg['val_trans'],
                           clip=cfg['val_clip'], aug=cfg['val_aug'],
                           sequence_mode=cfg['val_seq_mode'],
                           resize=cfg['resize'],
                           test=subj)
    test_set = ChaosDataset(ds, set_type='test', transform=cfg['test_trans'],
                            clip=cfg['test_clip'], aug=cfg['test_aug'],
                            sequence_mode=cfg['test_seq_mode'],
                            resize=cfg['resize'],
                            test=subj)
    if verbose:
        print('train: ', len(train_set))
        print('val: ', len(val_set))
        print('test: ', len(test_set))
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

    # Set device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        warnings.warn('Device is running on CPU, not GPU!')

    # Define model
    if cfg['model_arch'] == 'modular':
        model = pytorch_arch.GeneralUNet3D(cfg['n_layers'], cfg['in_channels'], cfg['model_cap'],
                                           cfg['out_channels_final'], cfg['channel_growth'],
                                           cfg['coord_conv'], cfg['transfer_layer']).to(device)

    # Set up adaptive loss if selected
    loss = None
    if loss_type == 'dice':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    else:
        raise NotImplementedError('Only Dice loss currently implemented')

    # Define optimizer
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                                 gamma=cfg['gamma'])

    if cfg['dry_run']:
        inputs, targets, names = next(iter(dataloaders['test']))
        print('test set info:')
        print('inputs', inputs.shape)
        print('targets', targets.shape)
        print('names', names)

        print('Model Summary:')
        # summary(model, input_size=(3, 224, 224))
        summary(model, input_size=(inputs.shape[1:]))
        return inputs, targets, names, None

    else:
        # Tensorboardx writer, model, config paths
        writer_dir = Path(output_path, 'tb_runs')
        config_dir = Path(output_path, 'config')
        model_dir = Path(output_path, 'trained_models', subj)
        writer_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(writer_dir)+f'/{model_version}_subj_{subj}')
        # Model graph is useless without additional tweaks to name layers appropriately
        # writer.add_graph(model, torch.zeros(1, 3, 256, 256).to(device), verbose=True)

        # Train Model
        model, best_loss = train_model(model, optimizer, exp_lr_scheduler, device, dataloaders,
                                       num_epochs=cfg['num_epochs'], tb_writer=writer,
                                       verbose=verbose, loss_func=loss)

        # Write outputs and save model
        cfg['best_loss'] = best_loss
        inputs, targets, masks, names = next(iter(dataloaders['test']))
        model.eval()
        model.to('cpu')
        model_pred = model(inputs)
        masked_target = targets.detach().numpy()*masks.numpy()
        masked_pred = model_pred.detach().numpy()*masks.numpy()
        test_mse = ((masked_target-masked_pred)**2).sum()/masks.numpy().sum()
        cfg['test_mse'] = test_mse
        masked_target = np.where(masked_target > 0, masked_target, np.nan)
        masked_pred = np.where(masked_pred > 0, masked_pred, np.nan)
        cfg['true_ave_stiff'] = np.nanmean(masked_target)
        cfg['test_ave_stiff'] = np.nanmean(masked_pred)

        config_file = Path(config_dir, f'{model_version}_subj_{subj}.pkl')
        with open(config_file, 'wb') as f:
            pkl.dump(cfg, f, pkl.HIGHEST_PROTOCOL)

        writer.close()
        torch.save(model.state_dict(), str(model_dir)+f'/model_{model_version}.pkl')
        return inputs, targets, masks, names, model


def process_kwargs(kwargs):
    cfg = default_cfg()
    for key in kwargs:
        val = str2bool(kwargs[key])
        cfg[key] = val
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


def default_cfg():
    cfg = {'train_trans': True, 'train_clip': True, 'train_aug': True, 'train_sample': 'shuffle',
           'val_trans': True, 'val_clip': True, 'val_aug': False, 'val_sample': 'shuffle',
           'test_trans': True, 'test_clip': True, 'test_aug': False,
           'train_seq_mode': 'random', 'val_seq_mode': 'random', 'test_seq_mode': 'random',
           'subj': '01', 'batch_size': 50, 'model_cap': 16, 'lr': 1e-2, 'step_size': 20,
           'gamma': 0.1, 'num_epochs': 40, 'dry_run': False, 'coord_conv': True, 'loss': 'dice',
           'model_arch': 'base', 'n_layers': 3, 'in_channels': 1, 'out_channels_final': 1,
           'channel_growth': False, 'transfer_layer': False,
           'resize': False}
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_path', type=str, help='Path to input data.',
                        default='/pghbio/dbmi/batmanlab/Data/MRE/')
    parser.add_argument('--data_file', type=str, help='Name of input pickle.',
                        default='mre_ds_preprocess_4_combomask.p')
    parser.add_argument('--output_path', type=str, help='Path to store outputs.',
                        default='/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data')
    parser.add_argument('--model_version', type=str, help='Name given to this set of configs'
                        'and corresponding model results.',
                        default='tmp')
    parser.add_argument('--verbose', type=bool, help='Verbose printouts.',
                        default=True)
    cfg = default_cfg()
    for key in cfg:
        val = str2bool(cfg[key])
        if type(val) is bool:
            parser.add_argument(f'--{key}', action='store', type=str2bool,
                                default=val)
        else:
            parser.add_argument(f'--{key}', action='store', type=type(val),
                                default=val)

    args = parser.parse_args()
    train_seg_model(**vars(args))
