import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import RandomSampler
import time
import copy
# from mre.plotting import hv_dl_vis
from mre.mre_datasets import MRETorchDataset
from robust_loss_pytorch import adaptive
import warnings
from datetime import datetime
from tqdm import tqdm_notebook
from tensorboardX import SummaryWriter


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
    # print(pred.shape)
    masked_mse = (((pred - target)**2)*mask).sum()/mask.ceil().sum()
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


def calc_loss(pred, target, mask, metrics, loss_func=None, pixel_weight=0.5):

    if loss_func is None:
        pixel_loss = masked_mse(pred, target, mask)
        # slice_loss = masked_mse_slice(pred, target, mask)
        subj_loss = masked_mse_subj(pred, target, mask)
        # loss = pixel_loss + subj_loss + slice_loss
        loss = pixel_weight*pixel_loss + (1-pixel_weight)*subj_loss
        # loss = pixel_loss + subj_loss
        # print('pixel_loss', pixel_loss)
        # print('subj_loss', subj_loss)
        # print('loss', loss)
    else:
        pass
        # resid = masked_resid(pred, target, mask)
        # loss = torch.sum(loss_func.lossfun(resid))/mask.ceil().sum()

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['pixel_loss'] += pixel_loss.data.cpu().numpy() * target.size(0)
    metrics['subj_loss'] += subj_loss.data.cpu().numpy() * target.size(0)
    # metrics['slice_loss'] += slice_loss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, device, dataloaders, num_epochs=25, tb_writer=None,
                verbose=True, loss_func=None, sls=False, pixel_weight=1):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e16
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
                                loss = calc_loss(outputs, labels, masks, metrics, loss_func,
                                                 pixel_weight)
                                loss.backward()
                                optimizer.step()
                            else:
                                def closure():
                                    return calc_loss(outputs, labels, masks, metrics, loss_func,
                                                     pixel_weight)
                                optimizer.step(closure)
                        else:
                            loss = calc_loss(outputs, labels, masks, metrics, loss_func,
                                             pixel_weight)
                    # accrue total number of samples
                    epoch_samples += inputs.size(0)

                if phase == 'train':
                    if sls:
                        pass
                    else:
                        scheduler.step()
                if verbose:
                    print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model if is it best
                if phase == 'val' and epoch_loss < best_loss:
                    if verbose:
                        print("updating best model floor")
                        print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                elif phase == 'val' and epoch_loss < best_loss*1.1:
                    if verbose:
                        print("saving best model (within 10%) ")
                    best_model_wts = copy.deepcopy(model.state_dict())

                if tb_writer:
                    tb_writer.add_scalar(f'loss_{phase}', epoch_loss, epoch)
                    tb_writer.add_scalar(f'pixel_loss_{phase}',
                                         metrics['pixel_loss']/epoch_samples, epoch)
                    tb_writer.add_scalar(f'subj_loss_{phase}',
                                         metrics['subj_loss']/epoch_samples, epoch)
                    # tb_writer.add_scalar(f'slice_loss_{phase}',
                    #                      metrics['slice_loss']/epoch_samples, epoch)
                    if loss_func is not None:
                        alpha = loss_func.alpha()[0, 0].detach().numpy()
                        scale = loss_func.scale()[0, 0].detach().numpy()
                        tb_writer.add_scalar(f'alpha', alpha, epoch)
                        tb_writer.add_scalar(f'scale', scale, epoch)
            if verbose:
                time_elapsed = time.time() - since
                print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        except KeyboardInterrupt:
            print('Breaking out of training early.')
            break
    if verbose:
        print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss


def add_predictions(ds, model, model_params, dims=2):
    '''Given a standard MRE dataset, a model, and the associated params, generate MRE predictions
    and load them into that dataset.'''
    model.eval()
    eval_set = MRETorchDataset(ds, set_type='eval', dims=dims)
    dataloader = DataLoader(eval_set, batch_size=16, shuffle=False, num_workers=2)
    for inputs, targets, masks, names in dataloader:
        prediction = model(inputs).data.cpu().numpy()
        if dims == 2:
            for i, name in enumerate(names):
                subj, z = name.split('_')
                z = int(z)
                ds['image_mre'].loc[{'subject': subj, 'z': z,
                                     'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
        elif dims == 3:
            for i, name in enumerate(names):
                ds['image_mre'].loc[{'subject': name,
                                     'mre_type': 'mre_pred'}] = (prediction[i, 0].T)**2
