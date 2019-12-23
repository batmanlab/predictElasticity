#!/usr/bin/env python

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from tensorboardX import SummaryWriter
from mre.segmentation import ChaosDataset
from mre import pytorch_arch_old
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
    cfg = process_kwargs(kwargs)
    torch.manual_seed(cfg['seed'])
    if verbose:
        print(cfg)
    ds = xr.open_dataset(Path(data_path, data_file))
    if verbose:
        print(ds)
    subj = cfg['subj']
    batch_size = cfg['batch_size']
    loss_type = cfg['loss']

    if cfg['train_seq_mode'] is None:
        cfg['train_seq_mode'] = cfg['def_seq_mode']
    if cfg['val_seq_mode'] is None:
        cfg['val_seq_mode'] = cfg['def_seq_mode']
    if cfg['test_seq_mode'] is None:
        cfg['test_seq_mode'] = cfg['def_seq_mode']

    # Start filling dataloaders
    dataloaders = {}
    train_set = ChaosDataset(ds, set_type='train', transform=cfg['train_trans'],
                             clip=cfg['train_clip'], aug=cfg['train_aug'],
                             sequence_mode=cfg['train_seq_mode'],
                             resize=cfg['resize'], model_arch=cfg['model_arch'],
                             test_subj=subj, seed=cfg['seed'], verbose=cfg['dry_run'])
    val_set = ChaosDataset(ds, set_type='val', transform=cfg['val_trans'],
                           clip=cfg['val_clip'], aug=cfg['val_aug'],
                           sequence_mode=cfg['val_seq_mode'],
                           resize=cfg['resize'], model_arch=cfg['model_arch'],
                           test_subj=subj, seed=cfg['seed'], verbose=cfg['dry_run'])
    test_set = ChaosDataset(ds, set_type='test', transform=cfg['test_trans'],
                            clip=cfg['test_clip'], aug=cfg['test_aug'],
                            sequence_mode=cfg['test_seq_mode'],
                            resize=cfg['resize'], model_arch=cfg['model_arch'],
                            test_subj=subj, seed=cfg['seed'], verbose=cfg['dry_run'])
    if verbose:
        print('train: ', len(train_set))
        print('val: ', len(val_set))
        print('test: ', len(test_set))
    if cfg['train_sample'] == 'shuffle':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                          num_workers=2)
    elif cfg['train_sample'] == 'resample':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                          sampler=RandomSampler(
                                              train_set, replacement=True,
                                              num_samples=cfg['train_num_samples']),
                                          num_workers=2),
    if cfg['val_sample'] == 'shuffle':
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                        num_workers=2)
    elif cfg['val_sample'] == 'resample':
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                        sampler=RandomSampler(
                                            val_set, replacement=True,
                                            num_samples=cfg['val_num_samples']),
                                        num_workers=2),
    dataloaders['test'] = DataLoader(test_set, batch_size=len(test_set), shuffle=False,
                                     num_workers=2)

    # Set device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        warnings.warn('Device is running on CPU, not GPU!')

    # Define model
    if cfg['model_arch'] == '3D':
        print('3d')
        model = pytorch_arch_old.GeneralUNet3D(cfg['n_layers'], cfg['in_channels'], cfg['model_cap'],
                                               cfg['out_channels_final'], cfg['channel_growth'],
                                               cfg['coord_conv'], cfg['transfer_layer'])
    elif cfg['model_arch'] == '2D':
        model = pytorch_arch_old.GeneralUNet2D(cfg['n_layers'], cfg['in_channels'], cfg['model_cap'],
                                               cfg['out_channels_final'], cfg['channel_growth'],
                                               cfg['coord_conv'], cfg['transfer_layer'])

    # Set up adaptive loss if selected
    loss = None
    if loss_type == 'dice':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    else:
        raise NotImplementedError('Only Dice loss currently implemented')

    # Define optimizer
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                                 gamma=cfg['gamma'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg['dry_run']:
        inputs, targets, names = next(iter(dataloaders['train']))
        print('test set info:')
        print('inputs', inputs.shape)
        print('targets', targets.shape)
        print('names', names)

        print('Model Summary:')
        # summary(model, input_size=(3, 224, 224))
        dry_size = list(inputs.shape[1:])
        # dry_size[0] = 1
        print(dry_size)
        summary(model, input_size=tuple(dry_size))
        # return inputs, targets, names, None
        return [dataloaders]

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
        model, best_loss, best_dice, best_bce = train_model_core(
            model, optimizer, exp_lr_scheduler, device, dataloaders, num_epochs=cfg['num_epochs'],
            tb_writer=writer, verbose=verbose, loss_func=loss)

        # Write outputs and save model
        cfg['best_loss'] = best_loss
        cfg['best_dice'] = best_dice
        cfg['best_bce'] = best_bce
        inputs, targets, names = next(iter(dataloaders['test']))
        inputs = inputs.to('cuda:0')
        targets = targets.to('cuda:0')
        model.eval()

        model_pred = torch.sigmoid(model(inputs[0:1, :]))
        test_dice = dice_loss(model_pred, targets[0:1, :])
        test_dice = test_dice.to('cpu')
        model_pred_t1_in = model_pred.cpu().detach()
        del model_pred
        torch.cuda.empty_cache()
        cfg['test_dice_t1_in'] = test_dice.item()

        model_pred = torch.sigmoid(model(inputs[1:2, :]))
        test_dice = dice_loss(model_pred, targets[1:2, :])
        test_dice = test_dice.to('cpu')
        model_pred_t1_out = model_pred.cpu().detach()
        del model_pred
        torch.cuda.empty_cache()
        cfg['test_dice_t1_out'] = test_dice.item()

        model_pred = torch.sigmoid(model(inputs[2:3, :]))
        test_dice = dice_loss(model_pred, targets[2:3, :])
        test_dice = test_dice.to('cpu')
        model_pred_t2 = model_pred.cpu().detach()
        del model_pred
        torch.cuda.empty_cache()
        cfg['test_dice_t2'] = test_dice.item()

        config_file = Path(config_dir, f'{model_version}_subj_{subj}.pkl')
        with open(config_file, 'wb') as f:
            pkl.dump(cfg, f, pkl.HIGHEST_PROTOCOL)

        writer.close()
        torch.save(model.state_dict(), str(model_dir)+f'/model_{model_version}.pkl')
        names = [names[0]+'_t1_in', names[1]+'_t1_out', names[2]+'_t2']
        return [inputs, targets, names,
                torch.cat([model_pred_t1_in, model_pred_t1_out, model_pred_t2])]


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
           'train_seq_mode': None, 'val_seq_mode': None, 'test_seq_mode': 'all', 'def_seq_mode':
           'random', 'seed': 100,
           'subj': '01', 'batch_size': 50, 'model_cap': 16, 'lr': 1e-2, 'step_size': 20,
           'gamma': 0.1, 'num_epochs': 40, 'dry_run': False, 'coord_conv': False, 'loss': 'dice',
           'model_arch': 'modular', 'n_layers': 3, 'in_channels': 1, 'out_channels_final': 1,
           'channel_growth': False, 'transfer_layer': False, 'bce_weight': 0.5,
           'resize': False}
    return cfg


def train_model_core(model, optimizer, scheduler, device, dataloaders, num_epochs=25,
                     loss_func='dice', bce_weight=0.5, tb_writer=None, verbose=True):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e16
    best_dice = 1e16
    best_bce = 1e16
    for epoch in range(num_epochs):
        try:
            if verbose:
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    for param_group in optimizer.param_groups:
                        if verbose:
                            print("LR", param_group['lr'])

                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                metrics = defaultdict(float)
                epoch_samples = 0

                # iterate through batches of data for each epoch
                for data in dataloaders[phase]:
                    inputs = data[0].to(device)
                    labels = data[1].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss, dice, bce  = calc_loss(outputs, labels,
                                                     metrics, bce_weight=bce_weight)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # accrue total number of samples
                    epoch_samples += inputs.size(0)
                if phase == 'train':
                    scheduler.step()

                if verbose:
                    print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples
                epoch_dice = metrics['dice'] / epoch_samples
                epoch_bce = metrics['bce'] / epoch_samples

                # deep copy the model if is it best
                if phase == 'val' and epoch_loss < best_loss:
                    if verbose:
                        print("saving best model")
                    best_loss = epoch_loss
                    best_dice = epoch_dice
                    best_bce = epoch_bce
                    best_model_wts = copy.deepcopy(model.state_dict())

                if tb_writer:
                    tb_writer.add_scalar(f'loss_{phase}', epoch_loss, epoch)
                    tb_writer.add_scalar(f'dice_{phase}', epoch_dice, epoch)
                    tb_writer.add_scalar(f'bce_{phase}', epoch_bce, epoch)
            if verbose:
                time_elapsed = time.time() - since
                print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        except KeyboardInterrupt:
            print('Breaking out of training early.')
            break
    if verbose:
        # print('Best val loss: {:4f}'.format(best_loss))
        print(f'Best val bce: {best_bce:.3f}, dice: {best_dice:.3f}, loss: {best_loss:.3f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_dice, best_bce


def calc_loss(pred, target, metrics, bce_weight=0.2):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss, dice, bce


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    if pred.ndim == 5:
        intersection = (pred * target).sum(dim=(2, 3, 4))

        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=(2, 3, 4)) +
                                                     target.sum(dim=(2, 3, 4)) + smooth)))
    else:
        intersection = (pred * target).sum(dim=(2, 3))

        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=(2, 3)) +
                                                     target.sum(dim=(2, 3)) + smooth)))

    return loss.mean()


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument(
        '--data_path', type=str, help='Path to input data.',
        default='/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/Train_Sets/MR/')
    parser.add_argument('--data_file', type=str, help='Name of input file.',
                        default='xarray_chaos.nc')
    parser.add_argument('--output_path', type=str, help='Path to store outputs.',
                        default='/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/')
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
