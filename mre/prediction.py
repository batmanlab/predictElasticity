import time
import copy
from collections import defaultdict
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from lmfit.models import LinearModel

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

    def forward(self, pred, target, mask):
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
        for i in range(ord_num):
            # K[:, i, :, :, :] = K[:, i, :, :, :] + i * torch.ones((N, D, H, W),
            #                                                      dtype=torch.int,
            #                                                      requires_grad=False).cuda()
            K[i, :] = K[i, :] + i * torch.ones(S, dtype=torch.int, requires_grad=False).cuda()

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        # one = torch.ones(pred[mask_1].size(), dtype=torch.int, requires_grad=False).cuda()

        self.loss = torch.sum(torch.log(torch.clamp(pred[mask_0], min=1e-8, max=1e8))) \
            + torch.sum(torch.log(torch.clamp(1 - pred[mask_1], min=1e-8, max=1e8)))

        # del K
        # del one
        # del mask_0
        # del mask_1

        # N = N * H * W * D
        self.loss /= (-S)  # negative
        return self.loss


def calc_loss(pred, target, mask, metrics, loss_func=None, pixel_weight=0.05):

    if loss_func is None or loss_func == 'l2':
        pixel_loss = masked_mse(pred, target, mask)
        # slice_loss = masked_mse_slice(pred, target, mask)
        subj_loss = masked_mse_subj(pred, target, mask)
        # loss = pixel_loss + subj_loss + slice_loss
        loss = pixel_weight*pixel_loss + (1-pixel_weight)*subj_loss
        # loss = pixel_loss + subj_loss
        # print('pixel_loss', pixel_loss)
        # print('subj_loss', subj_loss)
        # print('loss', loss)
        metrics['pixel_loss'] += pixel_loss.data.cpu().numpy() * target.size(0)
        metrics['subj_loss'] += subj_loss.data.cpu().numpy() * target.size(0)
    elif loss_func == 'ordinal':
        ord_loss = OrdLoss()
        # print('requires_grad', pred[0].requires_grad)
        # print('requires_grad', pred[1].requires_grad)
        loss = ord_loss(pred[1], target, mask)
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
                bins=None):
    if loss_func is None:
        loss_func = 'l2'
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
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            if not sls:
                                with torch.autograd.detect_anomaly():
                                    loss = calc_loss(outputs, labels, masks, metrics, loss_func,
                                                     pixel_weight)
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
                                             pixel_weight)
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
                    if phase == 'val' and epoch_loss < best_loss:
                        if verbose:
                            print("updating best model floor")
                            print("saving best model")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                    # elif phase == 'val' and epoch_loss < best_loss*1.05:
                    #     if verbose:
                    #         print("saving best model (within 5%) ")
                    #     best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    if phase == 'train' and epoch_loss < best_loss:
                        if verbose:
                            print("saving best model (training)")
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                if tb_writer:
                    tb_writer.add_scalar(f'loss_{phase}', epoch_loss, epoch)
                    if loss_func != 'ordinal':
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
        for phase in phases:
            for data in dataloaders[phase]:
                inputs = data[0].to(device)
                names = data[3]
                # print(names)
                if loss_func == 'ordinal':
                    prediction = model(inputs)[0].data.cpu().numpy()
                else:
                    prediction = model(inputs).data.cpu().numpy()
                # print(prediction.shape)
                for i, name in enumerate(names):
                    # print(name)
                    # print(prediction[i])
                    # print(prediction[i][0])
                    # print(prediction[i, 0])
                    # ds['image_mre'].loc[{'subject': name,
                    #                      'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
                    if loss_func == 'l2':
                        # ds['image_mre'].loc[{'subject': name,
                        #                      'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
                        ds['image_mre'].loc[{'subject': name,
                                             'mre_type': 'mre_pred'}] = (prediction[i, 0].T)*100
                    elif loss_func == 'ordinal':
                        print(bins)
                        if bins == 'uniform':
                            centers = [311.32407407, 907.97222222, 1504.62037037,
                                       2101.26851852, 2697.91666667, 3294.56481481, 3891.21296296,
                                       4487.86111111, 5084.50925926, 5681.15740741, 6277.80555556,
                                       6874.4537037, 7471.10185185, 8067.75, 8664.39814815,
                                       9261.0462963, 9857.69444444, 10454.34259259, 11050.99074074,
                                       11647.63888889, 12244.28703704, 12840.93518519,
                                       13437.58333333, 14034.23148148, 14630.87962963,
                                       15227.52777778, 15824.17592593, 16420.82407407,
                                       17017.47222222, 17614.12037037, 18210.76851852,
                                       18807.41666667, 19404.06481481, 20000.71296296,
                                       20597.36111111, 21194.00925926, 21790.65740741,
                                       22387.30555556, 22983.9537037, 23580.60185185, 24177.25,
                                       24773.89814815, 25370.5462963, 25967.19444444,
                                       26563.84259259, 27160.49074074, 27757.13888889,
                                       28353.78703704, 28950.43518519, 29547.08333333,
                                       30143.73148148, 30740.37962963, 31337.02777778,
                                       31933.67592593]
                        elif bins == 'blocks':
                            centers = [124.75, 302., 457., 634.5, 763., 857.,
                                       961., 1058.5, 1146., 1233.5, 1322., 1414.,
                                       1499., 1570., 1648., 1744.5, 1859.5, 2001.5,
                                       2427.5, 2910.5, 3150.5, 3349.5, 3534.5, 3696.5,
                                       3897.5, 4149.5, 4377.5, 4601., 4903.5, 5233.,
                                       5509., 5767.5, 6089., 6452.5, 6832.5, 7277.5,
                                       7748.5, 8278.5, 8863., 9366.5, 9855., 10453.5,
                                       11184., 11951.5, 12691., 13492., 14262.5, 14967.,
                                       15806., 16923., 18085.5, 19530., 21824., 27715.75]

                        subj_pred = prediction[i, 0].T.astype(int)
                        pred_transform = np.zeros_like(subj_pred)
                        it = np.nditer(subj_pred, flags=['multi_index'])
                        print('assigning')
                        while not it.finished:
                            # print(it.multi_index)
                            # print(it[0])
                            # print(centers)
                            # print(pred_transform)
                            pred_transform[it.multi_index] = centers[it[0]]
                            it.iternext()
                        ds['image_mre'].loc[{'subject': name,
                                             'mre_type': 'mre_pred'}] = pred_transform
                    else:
                        raise ValueError('Cannot save predictions due to unknown loss function'
                                         f' {loss_func}')
    del inputs
    del labels
    del masks
    del outputs
    torch.cuda.empty_cache()

    return model, best_loss


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


def get_linear_fit(ds, do_cor=False, make_plot=True, verbose=True, return_df=False):
    '''Generate a linear fit between the average stiffness values for the true and predicted MRE
    values.  Only consider pixels in the mask region.'''

    true = []
    pred = []
    for subj in ds.subject:
        true_mre_region = (ds.sel(subject=subj, mre_type='mre')['image_mre'].values *
                           ds.sel(subject=subj, mask_type='combo')['mask_mre'].values)
        true_mre_region = np.where(true_mre_region > 0, true_mre_region, np.nan)
        pred_mre_region = (ds.sel(subject=subj, mre_type='mre_pred')['image_mre'].values *
                           ds.sel(subject=subj, mask_type='combo')['mask_mre'].values)
        pred_mre_region = np.where(pred_mre_region > 0, pred_mre_region, np.nan)
        if do_cor:
            slope = ds.sel(subject=subj)['val_slope'].values
            intercept = ds.sel(subject=subj)['val_intercept'].values
            pred_mre_region = (pred_mre_region-intercept)/slope
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
        plt.show()

    if verbose:
        print(result.fit_report())
        print('R2:', 1 - result.residual.var() / np.var(df_results['predict']))
    if return_df:
        return df_results
    else:
        return result.params['slope'].value, result.params['intercept'].value


def add_val_linear_cor(ds_val, ds_test):
    slope, intercept = get_linear_fit(ds_val, False, False)
    ds_test['val_slope'] = (('subject'), np.asarray([slope]*len(ds_test.subject)))
    ds_test['val_intercept'] = (('subject'), np.asarray([intercept]*len(ds_test.subject)))
