import time
import copy
from collections import defaultdict
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from lmfit.models import LinearModel
import xarray as xr
from scipy import ndimage as ndi

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import RandomSampler
from tensorboardX import SummaryWriter

# from mre.plotting import hv_dl_vis
from mre.mre_datasets import MRETorchDataset
from robust_loss_pytorch import adaptive
from mre.ord_bins import get_ord_binning


def masked_L1(pred, target, mask):
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()

    masked_L1 = (torch.abs(pred - target)*mask).sum()/mask.ceil().sum()
    # print('ceil sum:', mask.ceil().sum())
    # print('sum:', mask.sum())

    return masked_L1


def masked_resid(pred, target, mask):
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()

    resid = (pred - target)*mask
    return torch.flatten(resid, start_dim=1)


def masked_mse(pred, target, mask):
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()
    # norm = (target*mask).sum()
    # N, C, D, H, W = pred.size()
    # ord_num = C
    # self.loss = 0.0
    # First mask the target and prediction based on the MRE Combo Mask
    # pred = pred.transpose(0, 1)[:, (mask.transpose(0, 1)[0, :]) > 0]
    # target = target.transpose(0, 1)[:, (mask.transpose(0, 1)[0, :]) > 0]
    # var = target.clone()
    # var[var == 0] = 1
    # norm = var.sum()
    # S = target.size()[1]
    # print(pred.shape)
    masked_mse = (((pred - target)**2)*mask).sum()/mask.ceil().sum()
    # masked_mse = ((pred - target)**2).sum()/S
    # print('ceil sum:', mask.ceil().sum())
    # print('sum:', mask.sum())

    return masked_mse


def full_mse(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    mse = ((pred - target)**2).sum()/torch.numel(pred)
    # masked_mse = ((pred - target)**2).sum()/S
    # print('ceil sum:', mask.ceil().sum())
    # print('sum:', mask.sum())

    return mse


def masked_mse_subj(pred, target, mask):
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()
    mask_pred = pred*mask
    mask_target = target*mask
    if len(pred.shape) == 5:
        mask_norm = mask.sum([1, 2, 3, 4])
        mask_pred_mean = mask_pred.sum([1, 2, 3, 4])/mask_norm
        mask_target_mean = mask_target.sum([1, 2, 3, 4])/mask_norm
    else:
        mask_norm = mask.sum([1, 2, 3])
        mask_pred_mean = mask_pred.sum([1, 2, 3])/mask_norm
        mask_target_mean = mask_target.sum([1, 2, 3])/mask_norm

    subj_mse = ((mask_pred_mean - mask_target_mean)**2).mean()
    return subj_mse


def masked_mse_slice(pred, target, mask):
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()
    mask_pred = pred*mask
    mask_target = target*mask
    if len(pred.shape) == 5:
        mask_norm = mask.sum([1, 3, 4])
        mask_pred_mean = mask_pred.sum([1, 3, 4])/mask_norm
        mask_target_mean = mask_target.sum([1, 3, 4])/mask_norm
    else:
        mask_norm = mask.sum([1, 3])
        mask_pred_mean = mask_pred.sum([1, 3])/mask_norm
        mask_target_mean = mask_target.sum([1, 3])/mask_norm

    slice_mse = ((mask_pred_mean - mask_target_mean)**2).mean()
    return slice_mse


def get_labels_sid(args, depth):
    if args.dataset == 'kitti':
        alpha = 0.001
        beta = 80.0
        K = 71.0
    elif args.dataset == 'nyu':
        alpha = 0.02
        beta = 10.0
        K = 68.0
    else:
        print('No Dataset named as ', args.dataset)

    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    if torch.cuda.is_available():
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()

    labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    if torch.cuda.is_available():
        labels = labels.cuda()
    return labels.int()


def get_depth_sid(args, labels):
    if args.dataset == 'kitti':
        min = 0.001
        max = 80.0
        K = 71.0
    elif args.dataset == 'nyu':
        min = 0.02
        max = 80.0
        K = 68.0
    else:
        print('No Dataset named as ', args.dataset)

    if torch.cuda.is_available():
        alpha_ = torch.tensor(min).cuda()
        beta_ = torch.tensor(max).cuda()
        K_ = torch.tensor(K).cuda()
    else:
        alpha_ = torch.tensor(min)
        beta_ = torch.tensor(max)
        K_ = torch.tensor(K)

    # print('label size:', labels.size())
    # depth = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * labels / K_)
    depth = alpha_ * (beta_ / alpha_) ** (labels / K_)
    # print(depth.size())
    return depth.float()


class OrdLoss(nn.Module):
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """

    def __init__(self):
        super(OrdLoss, self).__init__()
        self.loss = 0.0

    def forward(self, pred, target, mask, widths=None):
        """
        Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
        over the entire image domain:
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """

        # assert pred.dim() == target.dim()
        # invalid_mask = target < 0
        # target[invalid_mask] = 0
        # return torch.sum(pred)

        N, C, D, H, W = pred.size()
        ord_num = C
        self.loss = 0.0
        # First mask the target and prediction based on the MRE Combo Mask
        pred = pred.transpose(0, 1)[:, (mask.transpose(0, 1)[0, :]) > 0]
        target = target.transpose(0, 1)[:, (mask.transpose(0, 1)[0, :]) > 0]
        S = target.size()[1]
        # for k in range(ord_num):
        #     #     '''
        #     #     p^k_(w, h) = e^y(w, h, 2k+1) / [e^(w, h, 2k) + e^(w, h, 2k+1)]
        #     #     '''
        #     #
        #     #     '''
        #     #     对每个像素而言，
        #     #     如果k小于l(w, h), log(p_k)
        #     #     如果k大于l(w, h), log(1-p_k)
        #     #     希望分类正确的p_k越大越好
        #     #     '''
        #     print(k)
        #     p_k = pred[:, k, :, :, :]
        #     p_k = p_k.view(N, 1, D, H, W)
        #     mask_0 = (target >= k).detach()   # 分类正确
        #     mask_1 = (target < k).detach()  # 分类错误

        #     self.loss += torch.sum(torch.log(torch.clamp(p_k[mask_0], min=1e-7, max=1e7))) \
        #         + torch.sum(torch.log(torch.clamp(1 - p_k[mask_1], min=1e-7, max=1e7)))

        # print('ord_num = ', ord_num)

        # faster version
        # K = torch.zeros((N, C, D, H, W), dtype=torch.int, requires_grad=False).cuda()
        K = torch.zeros((C, S), dtype=torch.int, requires_grad=False).cuda()
        if widths is not None:
            weights = torch.zeros((C, S), dtype=torch.float, requires_grad=False).cuda()
        for i in range(ord_num):
            # K[:, i, :, :, :] = K[:, i, :, :, :] + i * torch.ones((N, D, H, W),
            #                                                      dtype=torch.int,
            #                                                      requires_grad=False).cuda()
            K[i, :] = K[i, :] + i * torch.ones(S, dtype=torch.int, requires_grad=False).cuda()
            if widths is not None:
                weights[i, :] = widths[i] * torch.ones(S, dtype=torch.float,
                                                       requires_grad=False).cuda()

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        # one = torch.ones(pred[mask_1].size(), dtype=torch.int, requires_grad=False).cuda()

        # print(pred.size())
        # print(pred)
        # print(pred[0:4, 0:10])
        # print(pred[mask_0].size())
        # print(pred[mask_1].size())
        if widths is not None:
            wnorm_0 = 1/torch.mean(weights[mask_0])
            wnorm_1 = 1/torch.mean(weights[mask_1])
            self.loss = wnorm_0*torch.sum(weights[mask_0]*torch.log(
                torch.clamp(pred[mask_0], min=1e-8, max=1e8))) \
                + wnorm_1*torch.sum(weights[mask_1]*torch.log(
                    torch.clamp(1 - pred[mask_1], min=1e-8, max=1e8)))
        else:
            self.loss = torch.sum(torch.log(torch.clamp(pred[mask_0], min=1e-8, max=1e8))) \
                + torch.sum(torch.log(torch.clamp(1 - pred[mask_1], min=1e-8, max=1e8)))

        # del K
        # del one
        # del mask_0
        # del mask_1

        # N = N * H * W * D
        self.loss /= (-S)  # negative
        return self.loss


def calc_loss(pred, target, mask, metrics, loss_func=None, pixel_weight=0.05, widths=None,
              wave=False):

    if not wave and (loss_func is None or loss_func == 'l2'):
        pixel_loss = masked_mse(pred, target, mask)
        subj_loss = masked_mse_subj(pred, target, mask)
        loss = pixel_weight*pixel_loss + (1-pixel_weight)*subj_loss
        metrics['pixel_loss'] += pixel_loss.data.cpu().numpy() * target.size(0)
        metrics['subj_loss'] += subj_loss.data.cpu().numpy() * target.size(0)
    elif wave:
        # do stiffness
        pixel_loss_stiff = masked_mse(pred[:, 0:1, :], target[:, 0:1, :], mask)
        pixel_loss_wave = full_mse(pred[:, 1:2, :], target[:, 1:2, :])
        loss = 0.5*pixel_loss_stiff + 0.5*pixel_loss_wave
        metrics['pixel_loss_stiff'] += pixel_loss_stiff.data.cpu().numpy() * target.size(0)
        metrics['pixel_loss_wave'] += pixel_loss_wave.data.cpu().numpy() * target.size(0)

    elif loss_func == 'ordinal':
        ord_loss = OrdLoss()
        # print('requires_grad', pred[0].requires_grad)
        # print('requires_grad', pred[1].requires_grad)
        loss = ord_loss(pred[1], target, mask, widths=widths)
        # loss = masked_mse(pred[1], target, mask)

    else:
        pass
        # resid = masked_resid(pred, target, mask)
        # loss = torch.sum(loss_func.lossfun(resid))/mask.ceil().sum()

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    # metrics['pixel_loss'] += pixel_loss.data.cpu().numpy() * target.size(0)
    # metrics['subj_loss'] += subj_loss.data.cpu().numpy() * target.size(0)
    # metrics['slice_loss'] += slice_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, device, dataloaders, num_epochs=25, tb_writer=None,
                verbose=True, loss_func=None, sls=False, pixel_weight=1, do_val=True, ds=None,
                bins=None, nbins=0, do_clinical=False, wave=False):
    widths = centers = 0
    if loss_func is None:
        loss_func = 'l2'
    elif loss_func == 'ordinal':
        _, centers, widths = get_ord_binning(bins, nbins)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e16
    if do_val:
        phases = ['train', 'val', 'test']
    else:
        phases = ['train', 'test']
    for epoch in range(num_epochs):
        try:
            if verbose:
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                since = time.time()

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    for param_group in optimizer.param_groups:
                        if verbose:
                            if sls:
                                print('sls auto LR')
                            else:
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
                    masks = data[2].to(device)
                    if do_clinical:
                        clinical = data[4].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if do_clinical:
                            outputs = model(inputs, clinical)
                        else:
                            outputs = model(inputs)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if not sls:
                                with torch.autograd.detect_anomaly():
                                    loss = calc_loss(outputs, labels, masks, metrics, loss_func,
                                                     pixel_weight, widths=widths, wave=wave)
                                    loss.backward()
                                    optimizer.step()
                            else:
                                def closure():
                                    pass
                                    # return calc_loss(outputs, labels, masks, metrics, loss_func,
                                    #                  pixel_weight)
                                optimizer.step(closure)
                        else:
                            loss = calc_loss(outputs, labels, masks, metrics, loss_func,
                                             pixel_weight, widths=widths, wave=wave)
                    # accrue total number of samples
                    epoch_samples += inputs.size(0)

                if phase == 'train':
                    if not sls:
                        scheduler.step()

                if verbose:
                    print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model if is it best
                if do_val:
                    if (phase == 'val') and (epoch_loss < best_loss) and (epoch == 0 or epoch > 50):
                        if verbose:
                            print("updating best model floor")
                            print("saving best model")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                    elif phase == 'val' and epoch_loss < best_loss*1.01 and epoch > 51:
                        if verbose:
                            print("saving best model (within 1%) ")
                        best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    if phase == 'train' and epoch_loss < best_loss:
                        if verbose:
                            print("saving best model (training)")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                if tb_writer:
                    tb_writer.add_scalar(f'loss_{phase}', epoch_loss, epoch)
                    if wave:
                        tb_writer.add_scalar(f'pixel_loss_stiff{phase}',
                                             metrics['pixel_loss_stiff']/epoch_samples, epoch)
                        tb_writer.add_scalar(f'pixel_loss_wave_{phase}',
                                             metrics['pixel_loss_wave']/epoch_samples, epoch)
                    elif loss_func != 'ordinal':
                        tb_writer.add_scalar(f'pixel_loss_{phase}',
                                             metrics['pixel_loss']/epoch_samples, epoch)
                        tb_writer.add_scalar(f'subj_loss_{phase}',
                                             metrics['subj_loss']/epoch_samples, epoch)
                    # tb_writer.add_scalar(f'slice_loss_{phase}',
                    #                      metrics['slice_loss']/epoch_samples, epoch)
                    # if loss_func is not None:
                    #     alpha = loss_func.alpha()[0, 0].detach().numpy()
                    #     scale = loss_func.scale()[0, 0].detach().numpy()
                    #     tb_writer.add_scalar(f'alpha', alpha, epoch)
                    #     tb_writer.add_scalar(f'scale', scale, epoch)
            if verbose:
                time_elapsed = time.time() - since
                print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        except KeyboardInterrupt:
            print('Breaking out of training early.')
            break
    if verbose:
        if do_val:
            print('Best val loss: {:4f}'.format(best_loss))
        else:
            print('Best training loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()   # Set model to evaluate mode
    # iterate through batches of data for each epoch
    if ds:
        print('converting prediction to correct units')

        if wave:
            ds_mem = xr.Dataset(
                {'image_mre': (['subject', 'mre_type', 'x', 'y', 'z'],
                               np.zeros((ds.subject.size, 3, ds.x.size, ds.y.size,
                                         ds.z.size), dtype=np.int16)),
                 'mask_mre': (['subject', 'mask_type', 'x', 'y', 'z'],
                              np.zeros((ds.subject.size, 1, ds.x.size, ds.y.size,
                                        ds.z.size), dtype=np.int16)),
                 },

                coords={'subject': ds.subject,
                        'mask_type': ['combo'],
                        'mre_type': ['mre', 'mre_pred', 'wave_pred'],
                        'x': ds.x,
                        'y': ds.y,
                        'z': ds.z
                        }
            )
        else:
            ds_mem = xr.Dataset(
                {'image_mre': (['subject', 'mre_type', 'x', 'y', 'z'],
                               np.zeros((ds.subject.size, 2, ds.x.size, ds.y.size,
                                         ds.z.size), dtype=np.int16)),
                 'mask_mre': (['subject', 'mask_type', 'x', 'y', 'z'],
                              np.zeros((ds.subject.size, 1, ds.x.size, ds.y.size,
                                        ds.z.size), dtype=np.int16)),
                 },

                coords={'subject': ds.subject,
                        'mask_type': ['combo'],
                        'mre_type': ['mre', 'mre_pred'],
                        'x': ds.x,
                        'y': ds.y,
                        'z': ds.z
                        }
            )
        print(ds_mem)
        # ds_mem = ds_mem.load()
        print('loaded data to mem')
        # print(ds_mem)
        for phase in phases:
            print(phase)
            for data in dataloaders[phase]:
                inputs = data[0].to(device)
                clinical = [None]*inputs.shape[0]
                if do_clinical:
                    clinical = data[4].to(device)
                names = data[3]
                # print(names)
                # print(prediction.shape)
                # print('looping over names')
                for i, name in enumerate(names):
                    # print('loading pred to mem')
                    if loss_func == 'ordinal':
                        prediction = model(inputs[i:i+1])[0].data.cpu().numpy()
                        # print(prediction.shape)
                    else:
                        prediction = model(inputs[i:i+1], clinical[i:i+1]).data.cpu().numpy()
                    # print(name)
                    # print(prediction[i])
                    # print(prediction[i][0])
                    # print(prediction[i, 0])
                    # ds['image_mre'].loc[{'subject': name,
                    #                      'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
                    if loss_func == 'l2':
                        if wave:
                            ds_mem['image_mre'].loc[{
                                'subject': name,
                                'mre_type': 'mre_pred'}] = (prediction[0, 0].T)*100
                            ds_mem['image_mre'].loc[{
                                'subject': name,
                                'mre_type': 'wave_pred'}] = (prediction[0, 1].T)*100
                        else:
                            ds_mem['image_mre'].loc[{
                                'subject': name,
                                'mre_type': 'mre_pred'}] = (prediction[0, 0].T)*100
                    elif loss_func == 'ordinal':
                        subj_pred = prediction[0, 0].T.astype(int)
                        pred_transform = np.ones_like(subj_pred)
                        it = np.nditer(subj_pred, flags=['multi_index'])
                        # print('iterating pred_trans')
                        while not it.finished:
                            # print(it.multi_index)
                            # print(it[0])
                            # print(centers)
                            # print(pred_transform)
                            pred_transform[it.multi_index] = centers[it[0]]
                            it.iternext()
                        ds_mem['image_mre'].loc[{'subject': name}] = pred_transform
                    else:
                        raise ValueError('Cannot save predictions due to unknown loss function'
                                         f' {loss_func}')
    del inputs
    del labels
    del masks
    del outputs
    torch.cuda.empty_cache()

    return model, best_loss, ds_mem


def add_predictions(ds, model, model_params, dims=2, inputs=None):
    '''Given a standard MRE dataset, a model, and the associated params, generate MRE predictions
    and load them into that dataset.'''
    if inputs is None:
        inputs = ['t1_pre_water', 't1_pre_in', 't1_pre_out', 't1_pre_fat', 't2', 't1_pos_0_water',
                  't1_pos_70_water', 't1_pos_160_water', 't1_pos_300_water']

    # model.eval()
    eval_set = MRETorchDataset(ds, set_type='eval', dims=dims, inputs=inputs)
    dataloader = DataLoader(eval_set, batch_size=4, shuffle=False, num_workers=2)
    for inputs, targets, masks, names in dataloader:
        prediction = model(inputs.to('cuda:0')).data.cpu().numpy()
        if dims == 2:
            for i, name in enumerate(names):
                subj, z = name.split('_')
                z = int(z)
                ds['image_mre'].loc[{'subject': subj, 'z': z,
                                     'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
        elif dims == 3:
            for i, name in enumerate(names):
                # ds['image_mre'].loc[{'subject': name,
                #                      'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
                ds['image_mre'].loc[{'subject': name,
                                     'mre_type': 'mre_pred'}] = (prediction[i, 0].T)*200


def get_linear_fit(ds, do_cor=False, make_plot=True, verbose=True, return_df=False, erode=0):
    '''Generate a linear fit between the average stiffness values for the true and predicted MRE
    values.  Only consider pixels in the mask region.'''

    true = []
    pred = []
    if do_cor:
        slope = np.mean(ds['val_slope'].values)
        intercept = np.mean(ds['val_intercept'].values)
        print(slope, intercept)
    for subj in ds.subject:
        mask = ds.sel(subject=subj, mask_type='combo')['mask_mre'].values
        if erode != 0:
            for i in range(mask.shape[-1]):
                mask[:, :, i] = ndi.binary_erosion(mask[:, :, i],
                                                   iterations=erode).astype(mask.dtype)
        true_mre_region = (ds.sel(subject=subj, mre_type='mre')['image_mre'].values * mask)
        true_mre_region = np.where(true_mre_region > 0, true_mre_region, np.nan)
        pred_mre_region = (ds.sel(subject=subj, mre_type='mre_pred')['image_mre'].values * mask)
        pred_mre_region = np.where(pred_mre_region > 0, pred_mre_region, np.nan)
        if do_cor:
            # slope = np.mean(ds.sel(subject=subj)['val_slope'].values)
            # intercept = np.mean(ds.sel(subject=subj)['val_intercept'].values)
            # print(slope, intercept)
            pred_mre_region = (pred_mre_region-intercept)/slope
            pred_mre_region = np.where(pred_mre_region > 0, pred_mre_region, 0)
        true.append(np.nanmean(true_mre_region))
        pred.append(np.nan_to_num(np.nanmean(pred_mre_region)))

    df_results = pd.DataFrame({'true': true, 'predict': pred, 'subject': ds.subject.values})
    df_results['fibrosis'] = np.where(df_results.true > 4000,
                                      'Severe Fibrosis', 'Mild Fibrosis')
    model = LinearModel()
    # params = model.guess(df_results['predict'], x=df_results['true'])
    params = model.make_params(slope=0.5, intercept=0)
    result = model.fit(df_results['predict'], params, x=df_results['true'])

    if make_plot:
        import matplotlib.pyplot as plt
        result.plot()
        plt.title('Mre_true vs Mre_pred')
        plt.ylim(0, 12000)
        plt.xlim(0, 12000)
        plt.xlabel('True MRE')
        plt.ylabel('Predicted MRE')

    if verbose:
        print(result.fit_report())
        print('R2:', 1 - result.residual.var() / np.var(df_results['predict']))
    if return_df:
        return df_results
    else:
        return result.params['slope'].value, result.params['intercept'].value


def add_val_linear_cor(ds_val, ds_test, erode=0):
    slope, intercept = get_linear_fit(ds_val, False, False, erode=erode)
    ds_test['val_slope'] = (('subject'), np.asarray([slope]*len(ds_test.subject)))
    ds_test['val_intercept'] = (('subject'), np.asarray([intercept]*len(ds_test.subject)))
    ds_test['erode'] = (('subject'), np.asarray([erode]*len(ds_test.subject)))
