#!/usr/bin/env python

import os
from pathlib import Path
from collections import OrderedDict
import warnings
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsummary import summary
from tensorboardX import SummaryWriter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from mre.mre_datasets import MREtoXr, MRETorchDataset
from mre.prediction import train_model, add_predictions, add_val_linear_cor
from mre import pytorch_arch_2d, pytorch_arch_3d
from robust_loss_pytorch import adaptive
from mre.pytorch_arch_deeplab import AlignedXception, DeepLab
from mre.pytorch_arch_debug import Debug

import sls


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
    print(Path(__file__).parent.absolute())
    # Load config and data
    cfg = process_kwargs(kwargs)
    if verbose:
        print(cfg)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    if cfg['patient_list']:
        files = [Path(data_path, 'xarray_'+i.strip()+'.nc') for i in open(cfg['patient_list'])]
        xr_maker = MREtoXr(from_file=files)
    else:
        xr_maker = MREtoXr(from_file=Path(data_path, data_file))
    # xr_maker = MREtoXr(from_file='/pghbio/dbmi/batmanlab/Data/MRE/XR/*.nc')
    ds = xr_maker.get_ds()
    # ds = ds.load()
    if verbose:
        print(ds)
    batch_size = cfg['batch_size']
    loss_type = cfg['loss']
    use_sls = False
    if cfg['lr_scheduler'] == 'sls':
        use_sls = True

    # Start filling dataloaders
    dataloaders = {}
    if cfg['subj'] is None:
        shuffle_list = np.asarray(ds.subject)
        np.random.shuffle(shuffle_list)
        train_idx = int(0.7*len(shuffle_list))
        val_idx = train_idx+int(0.2*len(shuffle_list))
        train_list = list(shuffle_list[:train_idx])
        val_list = list(shuffle_list[train_idx:val_idx])
        test_list = list(shuffle_list[val_idx:])
    elif cfg['sampling_breakdown'] == 'dumb':
        test_list = cfg['subj']
        shuffle_list = [subj for subj in ds.subject.values if subj not in test_list]
        shuffle_list = np.asarray(shuffle_list)
        np.random.shuffle(shuffle_list)
        if cfg['do_val']:
            train_idx = int(0.75*len(shuffle_list))
            train_list = list(shuffle_list[:train_idx])
            val_list = list(shuffle_list[train_idx:])
        else:
            train_list = list(shuffle_list)

    elif cfg['sampling_breakdown'] == 'smart_LOO':
        train_subj = ['1550', '1839', '0126', '0890', '1899', '1456', '1851', '0415', '0937',
                      '1829', '0173', '1083', '1561', '1795', '1033', '1123', '0659', '1504',
                      '0932', '1417', '1491', '1798', '0693', '0029', '1748', '1287', '2034',
                      '0655', '0954', '1103', '0491', '1603', '1843', '1791', '0975', '1311',
                      '1948', '1367', '1979', '1727', '0401', '1667', '0735', '1453', '0006',
                      '0734', '0898', '1793', '0612', '1940', '1699', '1883', '1526', '0461',
                      '0747', '1595', '1578', '1893', '1474', '0210', '1574', '1736', '0737',
                      '1400', '0628', '1106', '0509', '1722', '1530', '1896', '0556', '1435',
                      '1149', '1554', '1790', '0020', '1110', '0564', '1980', '1786', '0291',
                      '1144', '0872', '0931', '0344', '1715', '1590', '0704', '0830', '2007',
                      '1765', '1217', '1819', '1119', '1395', '2029', '0510', '1642', '2046',
                      '1714', '0235', '1789', '0929', '1045', '1447', '1935', '1541', '1853',
                      '1072', '1412', '0043']
        val_subj = ['1720', '1077', '1448', '1329', '1903', '1464', '1967', '1360', '0979', '1337',
                    '1529', '1341', '1336', '1121', '0860', '1706', '0914', '1679', '1076', '1712',
                    '0135', '1671', '0904', '1785', '1806', '0748', '1271', '0995', '2001', '1577',
                    '0234', '1694', '1404', '0940', '0653', '1382', '0492', '0219']

        test_list = cfg['subj']
        train_list = [subj for subj in train_subj if subj not in test_list]
        val_list = [subj for subj in val_subj if subj not in test_list]

    elif cfg['sampling_breakdown'] == 'smart':
        if cfg['do_older_dataset'] is True:
            # Needs to be hardcoded for now, problem with calc on the fly
            high_subj = ['1106', '1853', '0173', '1033', '0954', '1427', '2007', '1736', '1967',
                         '1474', '1343', '0135', '0890', '1296', '1839', '1395', '1526', '0838',
                         '1336', '1103', '0929', '1149', '1577', '0747', '2001', '1590', '1083',
                         '0932', '1530', '0291', '1790', '0210', '1785', '1574', '1896', '1789',
                         '1979', '1311', '1722', '0491', '1714', '1595', '1367', '1935', '0344',
                         '0931', '1798', '1287', '0659', '0234', '1715', '0126', '1271', '1791',
                         '1851', '0219', '1550', '0693', '0461', '1720', '2046', '1077', '0235',
                         '0898', '0628']
            low_subj = ['0737', '1426', '1712', '0995', '1464', '1123', '1400', '1278',
                        '1072', '0704', '1360', '1209', '0564', '1883', '1806', '1045',
                        '1417', '1404', '1893', '0655', '1699', '1028', '1144', '1554',
                        '1795', '1578', '0164', '1579', '0020', '1453', '1341', '1903',
                        '1679', '1447', '0006', '1344', '0904', '1215', '1456', '1671',
                        '1483', '1121', '0612', '1765', '0914', '1748', '0395', '1727',
                        '1940', '1948', '0415', '1110', '2034', '1217', '1603', '0734',
                        '1504', '0830', '0860', '0979', '1819', '1119', '1642', '1491',
                        '1694', '1843', '1433', '1529', '1706', '0872', '1541', '1561',
                        '0401', '1382', '1667', '1980', '1545', '0653', '1829', '0043',
                        '0975', '1076', '2029', '0556', '1435', '0735', '0029', '1303',
                        '0509', '1899', '1412', '1337', '1329', '1793', '0937', '1786',
                        '0492', '0940', '0748', '1448', '0510', '0989']
            test_list = cfg['subj']
            high_subj = [subj for subj in high_subj if subj not in test_list]
            low_subj = [subj for subj in low_subj if subj not in test_list]

            val_list = high_subj[:14] + low_subj[:24]
            train_list = high_subj[14:] + low_subj[24:]

        else:
            # Needs to be hardcoded for now, problem with calc on the fly
            high_subj = ['1736', '2001', '1935', '0898', '1149', '0931', '1590', '1033',
                         '0135', '1271', '1474', '1577', '1851', '1103', '0173', '1106',
                         '1790', '0747', '1979', '1077', '0954', '1530', '0291', '0210',
                         '1550', '1798', '1574', '0890', '1336', '1395', '1853', '1896',
                         '1967', '2046', '1722', '0491', '1714', '1715', '1526', '1720',
                         '0344', '0932', '1791', '1311', '0659', '0234', '1785', '0126',
                         '1287', '1839', '0219', '1595', '0693', '0461', '1789', '2007',
                         '1083', '0235', '0929', '0628', '1367']
            low_subj = ['0735', '1529', '1806', '1144', '1110', '1795', '1843', '2029',
                        '1712', '0020', '1883', '1699', '1076', '1786', '0653', '1448',
                        '0006', '1464', '1793', '1119', '2034', '1360', '1667', '1382',
                        '1400', '0940', '1329', '1679', '1578', '1727', '1893', '1899',
                        '1903', '1948', '1447', '0737', '1341', '0872', '0904', '1045',
                        '1453', '1404', '0655', '1765', '1694', '1121', '0734', '1706',
                        '0914', '0401', '1491', '0509', '1829', '1217', '1337', '0937',
                        '0492', '1456', '1541', '1642', '1554', '0415', '1748', '0704',
                        '1819', '0043', '0995', '1980', '1940', '0612', '1561', '0748',
                        '0029', '1412', '0556', '1504', '1435', '1417', '0975', '0510',
                        '0979', '0860', '1603', '0564', '1072', '1123', '0830', '1671']
            test_list = cfg['subj']
            high_subj = [subj for subj in high_subj if subj not in test_list]
            low_subj = [subj for subj in low_subj if subj not in test_list]

            val_list = high_subj[:10] + low_subj[:20]
            train_list = high_subj[10:] + low_subj[20:]

    elif cfg['sampling_breakdown'] == 'stratified':
        test_list = cfg['subj']
        df_strat = pd.read_pickle(
            '/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/MRE/df_strat_v0.pkl')
        # df_strat = df_strat.drop(index=test_list)
        # mskf = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=30,
        #                                         random_state=cfg['seed'])
        mskf = StratifiedShuffleSplit(n_splits=1, test_size=28, random_state=cfg['seed'])
        splits = mskf.split(df_strat.index.values, df_strat[['gender']].values.flatten())
        train_index, val_index = next(splits)
        print(val_index)
        train_list = df_strat.index[train_index].values
        val_list = df_strat.index[val_index].values
        train_list = [i for i in train_list if i not in test_list]
        val_list = [i for i in val_list if i not in test_list]
        # train_list = list(df_strat.index.values[train_list])
        # val_list = list(df_strat.index.values[val_list])

    elif cfg['sampling_breakdown'] == 'stratified_fixed':
        train_subj = ['1727', '1903', '2034', '1072', '1382', '0830', '0655', '1851', '1980',
                      '1839', '1554', '0234', '1967', '1149', '0020', '0653', '1935', '1360',
                      '0043', '1144', '1899', '1694', '1577', '0135', '1574', '1395', '1033',
                      '0937', '1404', '1578', '1456', '0556', '0126', '1474', '1789', '0219',
                      '1819', '1679', '1893', '0904', '0659', '1526', '1367', '1217', '1447',
                      '1504', '1541', '1720', '1529', '1699', '1671', '1714', '1329', '0510',
                      '0401', '1603', '0929', '0890', '1798', '1793', '1400', '1561', '1722',
                      '0734', '0693', '1642', '1979', '1795', '1271', '2046', '1948', '0461',
                      '0210', '0509', '0898', '1435', '0872', '0564', '1123', '0747', '1896',
                      '1712', '1448', '0006', '0975', '1464', '1336', '0173', '1843', '0932',
                      '0748', '1736', '1083', '0995', '1667', '1791', '1715', '1412', '0291',
                      '0704', '1853', '2001', '1341', '0029', '0931', '1790', '1076', '1786',
                      '1110', '1748', '1829', '1550', '1311', '0415', '0491', '0735', '0914']
        val_subj = ['2007', '1045', '0612', '1530', '1417', '0628', '0979', '1106', '1765', '1077',
                    '1806', '0235', '1287', '0492', '0344', '1337', '1883', '1453', '1491', '1785',
                    '0940', '2029', '1103', '0954', '0737', '0860', '1595', '1121', '1119', '1590',
                    '1706', '1940']

        test_list = cfg['subj']
        train_list = [subj for subj in train_subj if subj not in test_list]
        val_list = [subj for subj in val_subj if subj not in test_list]

    train_set = MRETorchDataset(ds.sel(subject=train_list), set_type='train', **cfg)
    cfg['norm_clin_vals'] = train_set.norm_clin_vals
    if cfg['do_val']:
        val_set = MRETorchDataset(ds.sel(subject=val_list), set_type='val', **cfg)
    test_set = MRETorchDataset(ds.sel(subject=test_list), set_type='test', **cfg)

    if verbose:
        print('train: ', len(train_set))
        if cfg['do_val']:
            print('val: ', len(val_set))
        print('test: ', len(test_set))
    num_workers = cfg['num_workers']

    if cfg['worker_init_fn'] == 'default':
        worker_init_fn = None
    elif cfg['worker_init_fn'] == 'rand_epoch':
        worker_init_fn = my_worker_init_fn
    else:
        raise ValueError('worker_init_fn specified incorrectly')

    if cfg['train_sample'] == 'shuffle':
        if cfg['norm'] == 'bn':
            drop_last = True
        else:
            drop_last = False
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, drop_last=drop_last,
                                          worker_init_fn=worker_init_fn)
    elif cfg['train_sample'] == 'resample':
        dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                          sampler=RandomSampler(
                                              train_set, replacement=True,
                                              num_samples=cfg['train_num_samples']),
                                          num_workers=num_workers,
                                          worker_init_fn=worker_init_fn)
    if cfg['do_val']:
        if cfg['val_sample'] == 'shuffle':
            dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                            num_workers=num_workers, drop_last=False,
                                            worker_init_fn=worker_init_fn)
        elif cfg['val_sample'] == 'resample':
            dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                            sampler=RandomSampler(
                                                val_set, replacement=True,
                                                num_samples=cfg['val_num_samples']),
                                            num_workers=num_workers,
                                            worker_init_fn=worker_init_fn)
    dataloaders['test'] = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, drop_last=False,
                                     worker_init_fn=worker_init_fn)

    # Set device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        warnings.warn('Device is running on CPU, not GPU!')

    # Define model
    if cfg['model_arch'] == 'base':
        raise NotImplementedError('"base" no longer valid model_arch.')
    elif cfg['model_arch'] == 'transfer':
        model = pytorch_arch_2d.PretrainedModel('name').to(device)
    elif cfg['model_arch'] == 'modular':
        if cfg['dims'] == 2:
            model = pytorch_arch_2d.GeneralUNet2D(cfg['n_layers'], cfg['in_channels'],
                                                  cfg['model_cap'], cfg['out_channels_final'],
                                                  cfg['channel_growth'], cfg['coord_conv'],
                                                  cfg['transfer_layer']).to(device)
        elif cfg['dims'] == 3:
            model = pytorch_arch_3d.GeneralUNet3D(cfg['n_layers'], cfg['in_channels'],
                                                  cfg['model_cap'], cfg['out_channels_final'],
                                                  cfg['channel_growth'], cfg['coord_conv'],
                                                  cfg['transfer_layer'], cfg['depth']).to(device)
    elif cfg['model_arch'] == 'deeplab':
        # model = DeepLabV3_3D(num_classes=cfg['out_channels_final'],
        #                      input_channels=cfg['in_channels'], resnet='resnet34_os8',
        #                      last_activation=None)
        if cfg['loss'] == 'ordinal':
            do_ord = True
        else:
            do_ord = False

        print(cfg['norm'])
        model = DeepLab(in_channels=cfg['in_channels'], out_channels=cfg['out_channels_final'],
                        output_stride=8, do_ord=do_ord, norm=cfg['norm'],
                        do_clinical=cfg['do_clinical'])
        if cfg['transfer']:

            # transfer_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data',
            #                      'trained_models', 'notebook',
            #                      'model_notebook_test_2020-02-20_12-04-13.pkl')
            # transfer_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data',
            #                      'trained_models', 'GROUP0',
            #                      'model_2020-03-05_13-50-59.pkl')
            transfer_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data',
                                 'trained_models', 'GROUP0',
                                 'model_2020-02-19_08-25-34_n0.pkl')

            # transfer_path = Path('/pghbio/dbmi/batmanlab/bpollack/predictElasticity/data/CHAOS/',
            #                      'trained_models', '001', 'model_2020-02-16_16-12-44.pkl')
            print('loading transfer')
            transfer_dict = torch.load(transfer_path, map_location='cuda:0')
            # print(transfer_dict.keys())
            transfer_dict = OrderedDict([(key[7:], val) for key, val in transfer_dict.items()])
            model_dict = model.state_dict()
            # print(model_dict.keys())
            # pretrained_dict = {k: v for k, v in transfer_dict.items() if k in model_dict}
            pretrained_dict = {k: v for k, v in transfer_dict.items() if (k in model_dict) and
                               (model_dict[k].shape == transfer_dict[k].shape)}
            # print(pretrained_dict.keys())
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            # print('freezing transfer params')
            # for name, param in model.named_parameters():
            #     if ((name in pretrained_dict.keys()) and
            #             (param.data.shape == pretrained_dict[name].shape)):
            #         param.requires_grad = False
    elif cfg['model_arch'] == 'debug':
        model = Debug(in_channels=cfg['in_channels'], out_channels=cfg['out_channels_final'])

    # Set up adaptive loss if selected
    loss_func = None
    if loss_type == 'robust':
        n_dims = train_set.target_images.shape[-1]*train_set.target_images.shape[-2]
        loss_func = adaptive.AdaptiveLossFunction(n_dims, np.float32, alpha_init=1.9, scale_lo=0.5)
        loss_params = torch.nn.ParameterList(loss_func.parameters())
        optimizer = optim.Adam(chain(model.parameters(), loss_params), lr=cfg['lr'])
    elif loss_type in ['l2', 'ordinal'] and cfg['lr_scheduler'] == 'step':
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif loss_type in ['l2', 'ordinal'] and cfg['lr_scheduler'] == 'cyclic':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr_max'], momentum=0.9,
                              weight_decay=cfg['weight_decay'])
    elif use_sls:
        optimizer = sls.Sls(model.parameters(),
                            n_batches_per_epoch=len(train_set)/float(cfg["batch_size"]))
    if loss_type == 'ordinal':
        loss_func = 'ordinal'

    # Define optimizer
    if cfg['lr_scheduler'] == 'step':
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'],
                                                     gamma=cfg['gamma'])
    elif cfg['lr_scheduler'] == 'cyclic':
        exp_lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['lr_min'],
                                                       max_lr=cfg['lr_max'],
                                                       cycle_momentum=False,
                                                       step_size_up=cfg['step_size'])
    elif use_sls:
        exp_lr_scheduler = None

    if torch.cuda.device_count() > 1 and not cfg['dry_run']:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, [0, 1])

    model.to(device)
    print('model loaded to gpu')

    if cfg['dry_run']:
        if cfg['do_clinical']:
            inputs, targets, masks, names, clinical = next(iter(dataloaders['test']))
        else:
            inputs, targets, masks, names = next(iter(dataloaders['test']))

        print('test set info:')
        print('inputs', inputs.shape)
        if cfg['do_clinical']:
            print('clinical', clinical.shape)
        print('targets', targets.shape)
        print('masks', masks.shape)
        print('names', names)

        print('Model Summary:')
        if cfg['do_clinical']:
            summary(model, input_size=[(cfg['in_channels'], 32, 256, 256), clinical.shape[1:]])
            return inputs, targets, masks, names, clinical, None
        else:
            summary(model, input_size=(cfg['in_channels'], 32, 256, 256))
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
        Path(xr_dir, 'test').mkdir(parents=True, exist_ok=True)
        Path(xr_dir, 'train').mkdir(parents=True, exist_ok=True)
        if cfg['do_val']:
            Path(xr_dir, 'val').mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(writer_dir)+f'/{model_version}_{subj_group}')
        # Model graph is useless without additional tweaks to name layers appropriately
        # writer.add_graph(model, torch.zeros(1, 3, 256, 256).to(device), verbose=True)

        # Train Model
        model, best_loss, ds_mem = train_model(model, optimizer, exp_lr_scheduler, device,
                                               dataloaders, num_epochs=cfg['num_epochs'],
                                               tb_writer=writer, verbose=verbose,
                                               loss_func=loss_func, sls=use_sls,
                                               pixel_weight=cfg['pixel_weight'],
                                               do_val=cfg['do_val'], ds=ds, bins=cfg['bins'],
                                               nbins=cfg['out_channels_final'],
                                               do_clinical=cfg['do_clinical'])
        print('model trained, handed off new mem_ds')

        # Write outputs and save model
        cfg['best_loss'] = best_loss
        if cfg['do_clinical']:
            inputs, targets, masks, names, clinical = next(iter(dataloaders['test']))
        else:
            inputs, targets, masks, names = next(iter(dataloaders['test']))
        inputs = inputs.to('cuda:0')
        targets = targets.to('cpu')
        masks.to('cpu')
        model.eval()
        # model.to('cpu')
        if cfg['loss'] == 'ordinal':
            model_pred = model(inputs)[0]
        elif cfg['do_clinical']:
            clinical = clinical.to('cuda:0')
            model_pred = model(inputs, clinical)
        else:
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
        # del model_pred
        # del masks
        # del targets
        # del inputs
        # torch.cuda.empty_cache()

        # add_predictions(ds, model, None, dims=cfg['dims'], inputs=cfg['inputs'])
        ds_mem['mask_mre'].loc[{'mask_type': 'combo'}] = ds['mask_mre'].sel(mask_type='combo')
        ds_mem['image_mre'].loc[{'mre_type': 'mre'}] = ds['image_mre'].sel(mre_type='mre')
        ds.close()
        ds_test = ds_mem.sel(subject=test_list)
        ds_train = ds_mem.sel(subject=train_list)
        if cfg['do_val']:
            ds_val = ds_mem.sel(subject=val_list)
            add_val_linear_cor(ds_val, ds_test, cfg['erode_mask'])
            ds_val_stub = ds_val.sel(mre_type='mre_pred')['image_mre']
            ds_val_stub.to_netcdf(Path(xr_dir, 'val', f'xarray_pred_{subj_group}.nc'))
            ds_val.close()
            ds_val_stub.close()
        ds_test_stub = ds_test.sel(mre_type='mre_pred')[['image_mre', 'val_slope', 'val_intercept',
                                                         'erode']]
        ds_test_stub.to_netcdf(Path(xr_dir, 'test', f'xarray_pred_{subj_group}.nc'))
        ds_test.close()
        ds_test_stub.close()
        ds_train_stub = ds_train.sel(mre_type='mre_pred')['image_mre']
        ds_train_stub.to_netcdf(Path(xr_dir, 'train', f'xarray_pred_{subj_group}.nc'))
        ds_train.close()
        ds_train_stub.close()

        # consider changing output to just ds?
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


def my_worker_init_fn(worker_id):
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed(torch.random.get_rng_state()[0].item() + worker_id)


def default_cfg():
    cfg = {'train_trans': True, 'train_clip': True, 'train_aug': True, 'train_sample': 'shuffle',
           'val_trans': True, 'val_clip': True, 'val_aug': False, 'val_sample': 'shuffle',
           'test_trans': True, 'test_clip': True, 'test_aug': False,
           'train_num_samples': 200, 'val_num_samples': 100,
           'train_smear': 'gaussian', 'val_smear': False, 'test_smear': False,
           'smear_amt': 3,
           'batch_size': 64, 'model_cap': 16, 'subj': None,
           'gamma': 0.1, 'num_epochs': 40, 'dry_run': False, 'coord_conv': False, 'loss': 'l2',
           'mask_trimmer': False, 'mask_mixer': 'mixed', 'target_max': None, 'target_bins': 100,
           'model_arch': 'modular', 'n_layers': 7, 'in_channels': 5, 'out_channels_final': 1,
           'channel_growth': False, 'transfer_layer': False, 'seed': 100,
           'worker_init_fn': 'rand_epoch',
           'resize': False, 'patient_list': False, 'num_workers': 0, 'lr_scheduler': 'step',
           'lr': 1e-2, 'lr_max': 1e-2, 'lr_min': 1e-4, 'step_size': 20, 'dims': 2,
           'pixel_weight': 1.0, 'depth': False, 'bins': 'none',
           'sampling_breakdown': 'smart', 'do_clinical': False, 'do_older_dataset': False,
           'norm_clinical': False, 'norm_clin_vals': None, 'erode_mask': 0,
           'do_val': True, 'norm': 'bn', 'transfer': False, 'weight_decay': 0.1,
           'inputs': ['t1_pre_water', 't1_pre_in', 't1_pre_out', 't1_pre_fat', 't2',
                      't1_pos_0_water', 't1_pos_70_water', 't1_pos_160_water', 't1_pos_300_water']}
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
        elif key == 'inputs':
            parser.add_argument(f'--{key}', nargs='*',
                                default=val)
        elif type(val) is bool:
            parser.add_argument(f'--{key}', action='store', type=str2bool,
                                default=val)
        else:
            parser.add_argument(f'--{key}', action='store', type=type(val),
                                default=val)

    args = parser.parse_args()
    print(args)
    train_model_full(**vars(args))
