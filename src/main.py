import numpy as np
import random
import torch

from datasets.TinyImageNetDataset import TinyImageNetDataset
from torch.utils.data import Dataset, DataLoader

import argparse

from train import ResNet18Trainer
#from test import ResNet18Test
from utils.configer import Configer


SEED = 1991
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True  # To have ~deterministic results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    torch.autograd.set_detect_anomaly(True)
    configer = Configer(args)
    if configer.get('phase') == 'train':
        model = ResNet18Trainer(configer)
        model.init_model()
        model.train()
    elif configer.get('phase') == 'test':
        #model = ResNet18Test(configer)
        model.init_model()
        model.test()