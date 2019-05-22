#! /usr/bin/env python3

from pathlib import Path
import warnings
import argparse
import pickle as pkl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from tensorboardX import SummaryWriter
from mre.prediction_v2 import MREDataset
from mre.prediction_v2 import train_model
from mre import pytorch_unet_tb


def train_model_full(data_path: str, data_file: str, output_path: str, model_version: str = 'tmp',
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
    ds = pkl.load(open(Path(data_path, data_file), 'rb'))
    if verbose:
        print(ds)
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

    # Set device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        warnings.warn('Device is running on CPU, not GPU!')

    # Define model
    model = pytorch_unet_tb.UNet(1, cap=cfg['model_cap']).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                                 gamma=cfg['gamma'])

    if cfg['dry_run']:
        inputs, targets, masks, names = next(iter(dataloaders['test']))
        print('test set info:')
        print('inputs', inputs.shape)
        print('targets', targets.shape)
        print('masks', masks.shape)
        print('names', names)

        print('Model Summary:')
        summary(model, input_size=(inputs.shape[1:]))

    else:
        # Tensorboardx writer and model paths
        writer_dir = Path(output_path, 'tb_runs')
        model_dir = Path(output_path, 'trained_models', subj)
        writer_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(writer_dir)+f'/{model_version}_subj_{subj}')

        # Train Model
        model = train_model(model, optimizer, exp_lr_scheduler, device, dataloaders,
                            num_epochs=cfg['num_epochs'], tb_writer=writer, verbose=verbose)

        # Write outputs and save model
        writer.close()
        model.to('cpu')
        torch.save(model.state_dict(), str(model_dir)+f'/model_{model_version}.pkl')


def process_kwargs(kwargs):
    cfg = default_cfg()
    for key in kwargs:
        cfg[key] = kwargs[key]
    return cfg


def default_cfg():
    cfg = {'train_trans': True, 'train_clip': True, 'train_aug': True, 'train_sample': 'shuffle',
           'val_trans': True, 'val_clip': True, 'val_aug': False, 'val_sample': 'shuffle',
           'test_trans': True, 'test_clip': True, 'test_aug': False,
           'subj': '162', 'batch_size': 50, 'model_cap': 16, 'lr': 1e-2, 'step_size': 20,
           'gamma': 0.1, 'num_epochs': 40, 'dry_run': False,
           }
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
        parser.add_argument(f'--{key}', action='store', type=type(cfg[key]),
                            default=cfg[key])

    args = parser.parse_args()
    train_model_full(**vars(args))
