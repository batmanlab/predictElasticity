#!/usr/bin/env python

import os
from pathlib import Path
import warnings
import argparse
import pickle as pkl
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from tensorboardX import SummaryWriter

from mre.mre_datasets import MREtoXr, MRETorchDataset
from mre.prediction import train_model, add_predictions
from mre import pytorch_arch
from robust_loss_pytorch import adaptive


def train_model_full(data_path: str, data_file: str, output_path: str, model_version: str = 'tmp',
                     subj_group: str = 'notebook',
                     verbose: str = True, **kwargs) -> None:
    '''Function to start training MRE model given a user-defined set of parameters.

    This function is intended to be imported and called in interactive sessions, from the command
    line, or by a slurm job submission script.  This function should be able to tweak any part of
    the model (or models) via a standardized config in order to easily document the impact of
    parameter settings.

    Args:
        data_path (str): Full path to location of data.
        data_file (str): Name (or wildcard names) of input netcdf files.
        output_path (str): Full path to output directory.
        model_version (str): Name of model.
        verbose (str): Print or suppress cout statements.

    Returns:
        None
    '''
    print(os.getcwd())
    # Load config and data
    cfg = process_kwargs(kwargs)
    if verbose:
        print(cfg)
    torch.manual_seed(cfg['seed'])

    if cfg['patient_list']:
        files = [Path(data_path, 'xarray_'+i.strip()+'.nc') for i in open(cfg['patient_list'])]
        xr_maker = MREtoXr(from_file=files)
    else:
        xr_maker = MREtoXr(from_file=Path(data_path, data_file))
    # xr_maker = MREtoXr(from_file='/pghbio/dbmi/batmanlab/Data/MRE/XR/*.nc')
    ds = xr_maker.get_ds()
    ds = ds.load()
    if verbose:
        print(ds)
    batch_size = cfg['batch_size']
    loss_type = cfg['loss']

    # Start filling dataloaders
    if cfg['subj'] is None:
        dataloaders = {}
        np.random.seed(cfg['seed'])
        shuffle_list = np.asarray(ds.subject)
        np.random.shuffle(shuffle_list)
        train_idx = int(0.7*len(shuffle_list))
        val_idx = train_idx+int(0.2*len(shuffle_list))
        train_list = list(shuffle_list[:train_idx])
        val_list = list(shuffle_list[train_idx:val_idx])
        test_list = list(shuffle_list[val_idx:])
    else:
        dataloaders = {}
        test_list = cfg['subj']
        np.random.seed(cfg['seed'])
        shuffle_list = [subj for subj in ds.subject.values if subj not in test_list]
        shuffle_list = np.asarray(shuffle_list)
        np.random.shuffle(shuffle_list)
        train_idx = int(0.75*len(shuffle_list))
        train_list = list(shuffle_list[:train_idx])
        val_list = list(shuffle_list[train_idx:])

    train_set = MRETorchDataset(ds.sel(subject=train_list), set_type='train', **cfg)
    val_set = MRETorchDataset(ds.sel(subject=val_list), set_type='val', **cfg)
    test_set = MRETorchDataset(ds.sel(subject=test_list), set_type='test', **cfg)

    if verbose:
        print('train: ', len(train_set))
        print('val: ', len(val_set))
        print('test: ', len(test_set))
    num_workers = cfg['num_workers']
    if cfg['train_sample'] == 'shuffle':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers)
    elif cfg['train_sample'] == 'resample':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                          sampler=RandomSampler(
                                              train_set, replacement=True,
                                              num_samples=cfg['train_num_samples']),
                                          num_workers=num_workers),
    if cfg['val_sample'] == 'shuffle':
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers)
    elif cfg['val_sample'] == 'resample':
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                        sampler=RandomSampler(
                                            val_set, replacement=True,
                                            num_samples=cfg['val_num_samples']),
                                        num_workers=num_workers),
    dataloaders['test'] = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)

    # Set device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        warnings.warn('Device is running on CPU, not GPU!')

    # Define model
    if cfg['model_arch'] == 'base':
        raise NotImplementedError('"base" no longer valid model_arch.')
    elif cfg['model_arch'] == 'transfer':
        model = pytorch_arch.PretrainedModel('name').to(device)
    elif cfg['model_arch'] == 'modular':
        if cfg['dims'] == 2:
            model = pytorch_arch.GeneralUNet2D(cfg['n_layers'], cfg['in_channels'],
                                               cfg['model_cap'], cfg['out_channels_final'],
                                               cfg['channel_growth'], cfg['coord_conv'],
                                               cfg['transfer_layer']).to(device)
        elif cfg['dims'] == 3:
            model = pytorch_arch.GeneralUNet3D(cfg['n_layers'], cfg['in_channels'],
                                               cfg['model_cap'], cfg['out_channels_final'],
                                               cfg['channel_growth'], cfg['coord_conv'],
                                               cfg['transfer_layer']).to(device)

    # Set up adaptive loss if selected
    loss = None
    if loss_type == 'robust':
        n_dims = train_set.target_images.shape[-1]*train_set.target_images.shape[-2]
        loss = adaptive.AdaptiveLossFunction(n_dims, np.float32, alpha_init=1.9, scale_lo=0.5)
        loss_params = torch.nn.ParameterList(loss.parameters())
        optimizer = optim.Adam(chain(model.parameters(), loss_params), lr=cfg['lr'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    # Define optimizer
    if cfg['lr_scheduler'] == 'step':
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                                     gamma=cfg['gamma'])
    else:
        exp_lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr_min'],
                                                       max_lr=cfg['lr_max'],
                                                       cycle_momentum=False,
                                                       step_size_up=cfg['step_size'])

    if torch.cuda.device_count() > 1 and not cfg['dry_run']:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg['dry_run']:
        inputs, targets, masks, names = next(iter(dataloaders['test']))
        print('test set info:')
        print('inputs', inputs.shape)
        print('targets', targets.shape)
        print('masks', masks.shape)
        print('names', names)

        print('Model Summary:')
        # summary(model, input_size=(3, 224, 224))
        summary(model, input_size=(inputs.shape[1:]))
        return inputs, targets, masks, names, None

    else:
        # Tensorboardx writer, model, config paths
        writer_dir = Path(output_path, 'tb_runs')
        config_dir = Path(output_path, 'config')
        xr_dir = Path(output_path, 'XR', model_version)
        model_dir = Path(output_path, 'trained_models', subj_group)

        writer_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        xr_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(writer_dir)+f'/{model_version}_{subj_group}')
        # Model graph is useless without additional tweaks to name layers appropriately
        # writer.add_graph(model, torch.zeros(1, 3, 256, 256).to(device), verbose=True)

        # Train Model
        model, best_loss = train_model(model, optimizer, exp_lr_scheduler, device, dataloaders,
                                       num_epochs=cfg['num_epochs'], tb_writer=writer,
                                       verbose=verbose, loss_func=loss)

        # Write outputs and save model
        cfg['best_loss'] = best_loss
        inputs, targets, masks, names = next(iter(dataloaders['test']))
        inputs = inputs.to('cuda:0')
        targets = targets.to('cpu')
        masks.to('cpu')
        model.eval()
        # model.to('cpu')
        model_pred = model(inputs)
        model_pred.to('cpu')
        masked_target = targets.detach().numpy()*masks.numpy()
        masked_pred = model_pred.detach().cpu().numpy()*masks.numpy()
        test_mse = ((masked_target-masked_pred)**2).sum()/masks.numpy().sum()
        cfg['test_mse'] = test_mse
        masked_target = np.where(masked_target > 0, masked_target, np.nan)
        masked_pred = np.where(masked_pred > 0, masked_pred, np.nan)
        cfg['true_ave_stiff'] = np.nanmean(masked_target)
        cfg['test_ave_stiff'] = np.nanmean(masked_pred)

        config_file = Path(config_dir, f'{model_version}_{subj_group}.pkl')
        with open(config_file, 'wb') as f:
            pkl.dump(cfg, f, pkl.HIGHEST_PROTOCOL)

        writer.close()
        torch.save(model.state_dict(), str(model_dir)+f'/model_{model_version}.pkl')

        ds = ds.sel(subject=test_list)
        add_predictions(ds, model, None)
        ds.to_netcdf(Path(xr_dir, f'xarray_{subj_group}.nc'))

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
    cfg = {'train_trans': True, 'train_clip': True, 'train_aug': False, 'train_sample': 'shuffle',
           'val_trans': True, 'val_clip': True, 'val_aug': False, 'val_sample': 'shuffle',
           'test_trans': True, 'test_clip': True, 'test_aug': False,
           'batch_size': 64, 'model_cap': 16, 'subj': None,
           'gamma': 0.1, 'num_epochs': 40, 'dry_run': False, 'coord_conv': False, 'loss': 'l2',
           'mask_trimmer': False, 'mask_mixer': 'mixed', 'target_max': None, 'target_bins': 100,
           'model_arch': 'modular', 'n_layers': 7, 'in_channels': 5, 'out_channels_final': 1,
           'channel_growth': False, 'transfer_layer': False, 'seed': 100,
           'resize': False, 'patient_list': False, 'num_workers': 0, 'lr_scheduler': 'step',
           'lr': 1e-2, 'lr_max': 1e-2, 'lr_min': 1e-4, 'step_size': 20}
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_path', type=str, help='Path to input data.',
                        default='/pghbio/dbmi/batmanlab/Data/MRE/')
    parser.add_argument('--data_file', type=str, help='Name of input pickle.',
                        default='*.nc')
    parser.add_argument('--output_path', type=str, help='Path to store outputs.',
                        default='/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data')
    parser.add_argument('--model_version', type=str, help='Name given to this set of configs'
                        'and corresponding model results.',
                        default='tmp')
    parser.add_argument('--subj_group', type=str, help='Name of the held-out subject group.',
                        default='notebook')
    parser.add_argument('--verbose', type=bool, help='Verbose printouts.',
                        default=True)
    cfg = default_cfg()
    for key in cfg:
        val = str2bool(cfg[key])
        if key == 'subj':
            parser.add_argument(f'--{key}', nargs='*',
                                default=val)
        elif type(val) is bool:
            parser.add_argument(f'--{key}', action='store', type=str2bool,
                                default=val)
        else:
            parser.add_argument(f'--{key}', action='store', type=type(val),
                                default=val)

    args = parser.parse_args()
    train_model_full(**vars(args))
