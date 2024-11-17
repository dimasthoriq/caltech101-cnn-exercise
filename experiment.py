import os
import torch
from train import Trainer, evaluate
from models import CustomCNN, get_efficient_net, get_resnet
from utils import get_model_size, get_loaders


config = {
    # Model
    'archi': 'custom',
    'normalization': 'batch',
    'compression': None,

    # Training designs
    'dropout_rate': 0.,
    'weight_decay': 2e-5,
    'loss': 'CE',
    'from_scratch': 'imagenet',
    'learning_rate': 1e-1,
    'batch_size': 32,

    # Dataset
    'resize_size': 256,
    'crop_size': 224,
    'shuffle_train': True,

    # Training utils
    'max_epochs': 50,
    'print_freq': 5,
    'patience': 10,
    'min_delta': 1e-6,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

"""
archi: ['custom', 'efficientnet_v2_s', 'resnet101']
normalization: ['batch', 'layer', 'group']
compression: [None, 'SVD']
dropout_rate: [0., 0.2]
weight_decay: [2e-5, 0.]
loss: ['ce', 'squared_hinge']
from_scratch: ['imagenet', 'random', 'tune']
learning_rate: [1e-1, 1e-2, 1e-4]
batch_size: [32, 16, 8]
"""

if config['archi'] == 'efficientnet_v2_s':
    config['resize_size'] = 384
    config['crop_size'] = 384

else:
    config['resize_size'] = 256
    config['crop_size'] = 224
