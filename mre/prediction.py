import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import time
import copy

# need data to be ordered thusly:
# image_sequence,width,hight,depth


class MREDataset(Dataset):
    def __init__(self, xa_ds, set_type='train', transform=None, clip=False, seed=100):
        # inputs = ['T1Pre', 'T1Pos', 'T2SS', 'T2FR']
        inputs = ['T1Pre', 'T1Pos', 'T2SS']
        targets = ['elast']
        masks = ['comboMsk']

        np.random.seed(seed)
        shuffle_list = np.asarray(xa_ds.subject)
        np.random.shuffle(shuffle_list)

        if set_type == 'test':
            # input_set = xa_ds.subject_2d[20:]
            input_set = list(shuffle_list[0:2])
        elif set_type == 'val':
            # input_set = xa_ds.subject_2d[2:20]
            input_set = list(shuffle_list[2:9])
        elif set_type == 'train':
            # input_set = xa_ds.subject_2d[:2]
            input_set = list(shuffle_list[9:])
        else:
            raise AttributeError('Must choose one of ["train", "val", "test"] for `set_type`.')

        # pick correct input set
        xa_ds = xa_ds.sel(subject=input_set)

        # stack subject and z-slices to make 4 2D image groups for each 3D image group
        xa_ds = xa_ds.stack(subject_2d=('subject', 'z')).reset_index('subject_2d')
        subj_2d_coords = [f'{i.subject.values}_{i.z.values}' for i in xa_ds.subject_2d]
        xa_ds = xa_ds.assign_coords(subject_2d=subj_2d_coords)
        print(xa_ds)
        self.name_dict = dict(zip(range(len(subj_2d_coords)), subj_2d_coords))

        self.input_images = xa_ds.sel(sequence=inputs).transpose(
            'subject_2d', 'sequence', 'y', 'x').image.values
        self.target_images = xa_ds.sel(sequence=targets).transpose(
            'subject_2d', 'sequence', 'y', 'x').image.values
        self.mask_images = xa_ds.sel(sequence=masks).transpose(
            'subject_2d', 'sequence', 'y', 'x').image.values
        self.transform = transform
        self.clip = clip
        self.names = xa_ds.subject_2d.values

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        target = self.target_images[idx]
        if self.clip:
            image = np.where(image >= 700, 700, image)
            target = np.float32(np.digitize(target, list(range(0, 20000, 200))+[1e6]))
        mask = self.mask_images[idx]
        if self.transform:
            rot_angle = np.random.uniform(-5, 5, 1)
            # rot_angle = 45
            translations = np.random.uniform(-10, 10, 2)
            scale = np.random.uniform(0.9, 1.1, 1)
            image = self.input_transform(image, rot_angle, translations, scale)
            mask = self.affine_transform(mask[0], rot_angle, translations, scale)
            target = self.affine_transform(target[0], rot_angle, translations, scale)

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


def masked_mse(pred, target, mask):
    pred = pred.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()
    masked_mse = (((pred - target)*mask)**2).sum()/mask.sum()
    return masked_mse


def calc_loss(pred, target, mask, metrics, bce_weight=0.5):

    # bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = F.sigmoid(pred)
    # dice = dice_loss(pred, target)

    # loss = bce * bce_weight + dice * (1 - bce_weight)
    # loss = F.mse_loss(pred, target)
    loss = masked_mse(pred, target, mask)

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    # metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, device, dataloaders, num_epochs=25, tb_writer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e16
    total_iter = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            metrics = defaultdict(float)
            epoch_samples = 0

            # for inputs, labels, masks in dataloaders[phase]:
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
                    loss = calc_loss(outputs, labels, masks, metrics)
                    if tb_writer:
                        tb_writer.add_scalar(f'loss_{phase}', loss, total_iter)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
                total_iter += 1
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
