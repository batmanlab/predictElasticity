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

from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import SimpleITK as sitk
import panel as pn
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from mre.preprocessing import MRIImage
from lmfit.models import LinearModel
