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
from mre import pytorch_unet_tb
# from mre.plotting import hv_dl_vis
from mre.mre_datasets import MRETorchDataset
from robust_loss_pytorch import adaptive
import warnings
from datetime import datetime
from tqdm import tqdm_notebook
from tensorboardX import SummaryWriter


# need data to be ordered thusly:
# image_sequence,width,hight,depth


class MREDataset(Dataset):
    def __init__(self, xa_ds, set_type='train', transform=None, clip=False, seed=100, test='162',
                 aug=True, mask_mixer='mixed', mask_trimmer=False,
                 target_max=None, target_bins=100, resize=False):
        # inputs = ['T1Pre', 'T1Pos', 'T2SS', 'T2FR']
        inputs = ['T1Pre', 'T1Pos', 'T2SS']
        targets = ['elast']
        masks = ['comboMsk']

        xa_ds_test = xa_ds.sel(subject=[test])
        xa_ds = xa_ds.drop(test, dim='subject')
        np.random.seed(seed)
        shuffle_list = np.asarray(xa_ds.subject)
        np.random.shuffle(shuffle_list)

        if set_type == 'test':
            input_set = list(test)
        elif set_type == 'val':
            # input_set = xa_ds.subject_2d[2:20]
            input_set = list(shuffle_list[0:8])
        elif set_type == 'train':
            # input_set = xa_ds.subject_2d[:2]
            input_set = list(shuffle_list[8:])
        else:
            raise AttributeError('Must choose one of ["train", "val", "test"] for `set_type`.')

        # pick correct input set
        if set_type == 'test':
            xa_ds = xa_ds_test
        else:
            xa_ds = xa_ds.sel(subject=input_set)

        # stack subject and z-slices to make 4 2D image groups for each 3D image group
        xa_ds = xa_ds.stack(subject_2d=('subject', 'z')).reset_index('subject_2d')
        subj_2d_coords = [f'{i.subject.values}_{i.z.values}' for i in xa_ds.subject_2d]
        xa_ds = xa_ds.assign_coords(subject_2d=subj_2d_coords)

        # Remove data that doesn't have enough target pixels
        if mask_trimmer:
            bad_subj2d = []
            for subj2d in xa_ds.subject_2d.values:
                tmp_msk = xa_ds.sel(subject_2d=subj2d, sequence=masks).image
                mask_val = np.max(np.unique(tmp_msk))
                if tmp_msk.where(tmp_msk >= mask_val).sum()/mask_val < 1000:
                    bad_subj2d.append(subj2d)
            xa_ds = xa_ds.drop(bad_subj2d, dim='subject_2d')
        self.name_dict = dict(zip(range(len(xa_ds.subject_2d)), xa_ds.subject_2d.values))

        self.input_images = xa_ds.sel(sequence=inputs).transpose(
            'subject_2d', 'sequence', 'y', 'x').image.values
        self.target_images = xa_ds.sel(sequence=targets).transpose(
            'subject_2d', 'sequence', 'y', 'x').image.values
        self.mask_images = xa_ds.sel(sequence=masks).transpose(
            'subject_2d', 'sequence', 'y', 'x').image.values
        self.transform = transform
        self.aug = aug
        self.clip = clip
        self.names = xa_ds.subject_2d.values
        self.mask_mixer = mask_mixer
        self.target_max = target_max
        self.target_bins = target_bins
        self.resize = resize

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        mask = self.mask_images[idx]
        if self.mask_mixer == 'mixed':
            pass
        elif self.mask_mixer == 'intersection':
            mask = np.where(mask >= 0.5, 1.0, 0.0).astype(mask.dtype)
        elif self.mask_mixer == 'union':
            mask = np.where(mask > 0, 1.0, 0.0).astype(mask.dtype)

        image = self.input_images[idx]
        target = self.target_images[idx]
        if self.clip:
            image[0, :, :]  = np.where(image[0, :, :] >= 700, 700, image[0, :, :])
            image[1, :, :]  = np.where(image[1, :, :] >= 1250, 1250, image[1, :, :])
            image[2, :, :]  = np.where(image[2, :, :] >= 600, 600, image[2, :, :])
            if self.target_max is None:
                target = np.float32(np.digitize(target, list(range(0, 20000, 200))+[1e6]))
            else:
                target = np.where(target >= self.target_max, self.target_max, target)
                spacing = int(self.target_max/self.target_bins)
                cut_points = list(range(0, self.target_max, spacing)) + [1e6]
                target = np.float32(np.digitize(target, cut_points))

        if self.transform:
            if self.aug:
                rot_angle = np.random.uniform(-4, 4, 1)
                translations = np.random.uniform(-5, 5, 2)
                scale = np.random.uniform(0.95, 1.05, 1)
            else:
                rot_angle = 0
                translations = (0, 0)
                scale = 1
            image = self.input_transform(image, rot_angle, translations, scale)
            mask = self.affine_transform(mask[0], rot_angle, translations, scale)
            target = self.affine_transform(target[0], rot_angle, translations, scale)

        if self.resize:
            image_0 = transforms.ToPILImage()(image.numpy()[0])
            image_0 = transforms.functional.resize(image_0, (224, 224))
            image_0 = transforms.ToTensor()(image_0)
            image_1 = transforms.ToPILImage()(image.numpy()[1])
            image_1 = transforms.functional.resize(image_1, (224, 224))
            image_1 = transforms.ToTensor()(image_1)
            image_2 = transforms.ToPILImage()(image.numpy()[2])
            image_2 = transforms.functional.resize(image_2, (224, 224))
            image_2 = transforms.ToTensor()(image_2)
            image = torch.cat((image_0, image_1, image_2))
            target = transforms.ToPILImage()(target.numpy()[0])
            target = transforms.Resize((224, 224))(target)
            target = transforms.ToTensor()(target)
            mask = transforms.ToPILImage()(mask.numpy()[0])
            mask = transforms.Resize((224, 224))(mask)
            mask = transforms.ToTensor()(mask)

        image = torch.Tensor(image)
        target = torch.Tensor(target)
        mask = torch.Tensor(mask)

        return [image, target, mask, self.names[idx]]

    def affine_transform(self, input_slice, rot_angle=0, translations=0, scale=1):
        input_slice = transforms.ToPILImage()(input_slice)
        input_slice = TF.affine(input_slice, angle=rot_angle,
                                translate=list(translations), scale=scale, shear=0)
        input_slice = transforms.ToTensor()(input_slice)
        return input_slice

    def input_transform(self, input_image, rot_angle=0, translations=0, scale=1):

        # normalize and offset image
        image = input_image
        image = np.where(input_image <= 1e-9, np.nan, input_image)
        mean = np.nanmean(image, axis=(1, 2))
        std = np.nanstd(image, axis=(1, 2))
        image = ((image.T - mean)/std).T + 4
        image = np.where(image != image, 0, image)

        # perform affine transfomrations
        image0 = self.affine_transform(image[0], rot_angle, translations, scale)
        image1 = self.affine_transform(image[1], rot_angle, translations, scale)
        image2 = self.affine_transform(image[2], rot_angle, translations, scale)
        return torch.cat((image0, image1, image2))


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


def calc_loss(pred, target, mask, metrics, loss_func=None):

    if loss_func is None:
        loss = masked_mse(pred, target, mask)
    else:
        resid = masked_resid(pred, target, mask)
        loss = torch.sum(loss_func.lossfun(resid))/mask.ceil().sum()

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, device, dataloaders, num_epochs=25, tb_writer=None,
                verbose=True, loss_func=None):
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
                        loss = calc_loss(outputs, labels, masks, metrics, loss_func)
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

                # deep copy the model if is it best
                if phase == 'val' and epoch_loss < best_loss:
                    if verbose:
                        print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                if tb_writer:
                    tb_writer.add_scalar(f'loss_{phase}', epoch_loss, epoch)
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


def gen_LOO_models(ds, save_dir, trans=True, clip=True, cap=16, version=None, verbose=False):
    if version is None:
        version = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(version)

    # for subj in tqdm_notebook(['162'], desc='Subject'):
    for subj in tqdm_notebook(ds.coords['subject'].values, desc='Subject'):
        train_set = MREDataset(ds, set_type='train', transform=trans, clip=clip, test=subj)
        val_set = MREDataset(ds, set_type='val', transform=trans, clip=clip, test=subj)
        test_set = MREDataset(ds, set_type='test', transform=trans, clip=clip, test=subj)

        batch_size = 50
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                sampler=RandomSampler(train_set, replacement=True, num_samples=200),
                                num_workers=0),

            'val': DataLoader(val_set, batch_size=batch_size, shuffle=True,
                              num_workers=0),

            'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
        }

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            warnings.warn('Device is running on CPU, not GPU!')
        model = pytorch_unet_tb.UNet(1, cap=cap).to(device)

        optimizer_ft = optim.Adam(model.parameters(), lr=1e-2)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
        writer = SummaryWriter(f'runs/{version}_subj_{subj}')
        model = train_model(model, optimizer_ft, exp_lr_scheduler, device, dataloaders,
                            num_epochs=45, tb_writer=writer, verbose=verbose)
        writer.close()
        model.to('cpu')
        torch.save(model.state_dict(), save_dir+f'/{subj}/model_{version}.pkl')

        # model.eval()   # Set model to evaluate mode
        # test_loader = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=0,
        #                          verbose=verbose)
        # inputs, labels, masks, names = next(iter(test_loader))
        # model_pred = model(inputs)
        # hv_dl_vis(inputs, labels, masks, names, model_pred)


def add_LOO_predictions(ds, path='/pghbio/dbmi/batmanlab/Data/MRE/', version='2019-05-13_14-53-50',
                        extra_name='extra3', mask_trimmer=False, mask_mixer='mixed', cap=12,
                        coord_conv=False):

    for subj in tqdm_notebook(ds.coords['subject'].values, desc='Subject'):
        test_set = MREDataset(ds, set_type='test', transform=True, clip=True, test=subj, aug=False,
                              mask_mixer=mask_mixer, mask_trimmer=mask_trimmer)
        test_dl = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

        model_path = path+f'/{subj}/model_{version}.pkl'
        model = pytorch_unet_tb.UNet(1, cap=cap, coord_conv=coord_conv)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()

        for data in test_dl:
            # print(data)
            pred = model(data[0]).data.numpy()
            z_slice = data[-1][0].split('_')[-1]

            ds['image'].loc[dict(sequence=extra_name,
                                 subject=subj, z=int(z_slice))] = pred[0, 0, :, :]*(200)-100

    new_sequence = [a.replace(extra_name, 'mre_pred') for a in ds.sequence.values]
    ds = ds.assign_coords(sequence=new_sequence)
    return ds


def add_predictions(ds, model, model_params):
    '''Given a standard MRE dataset, a model, and the associated params, generate MRE predictions
    and load them into that dataset.'''
    model.eval()
    eval_set = MRETorchDataset(ds, set_type='eval')
    dataloader = DataLoader(eval_set, batch_size=16, shuffle=False, num_workers=2)
    for inputs, targets, masks, names in dataloader:
        prediction = model(inputs).data.cpu().numpy()
        for i, name in enumerate(names):
            subj, z = name.split('_')
            z = int(z)
            ds['image_mre'].loc[{'subject': subj, 'z': z,
                                 'mre_type': 'mre_pred'}] = prediction[i, 0].T*1000
