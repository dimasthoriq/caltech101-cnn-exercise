import os
import torch
from train import Trainer, evaluate, SquaredHingeLoss
from models import CustomCNN, get_efficient_net, get_resnet
from utils import get_model_size, get_loaders


config = {
    # Model
    'archi': 'custom',
    'normalization': 'batch',
    'compression': None,

    # Training designs
    'dropout_rate': 0.2,
    'weight_decay': 2e-5,
    'loss': 'ce',
    'from_scratch': 'tune',
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
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 10
}

"""
archi: ['custom', 'efficientnet_v2_s', 'resnet101']
normalization: ['batch', 'layer', 'group']
compression: [None, 128]
dropout_rate: [0., 0.2]
weight_decay: [2e-5, 0.]
loss: ['ce', 'squared_hinge']
from_scratch: ['imagenet', 'random', 'tune']
learning_rate: [1e-1, 1e-2, 1e-4]
batch_size: [32, 16, 8]
"""


def experiment(config):
    if config['archi'] == 'efficientnet_v2_s':
        config['resize_size'] = 384
        config['crop_size'] = 384
        model = get_efficient_net(from_scratch=config['from_scratch'])

    else:
        config['resize_size'] = 256
        config['crop_size'] = 224

        if config['archi'] == 'resnet101':
            model = get_resnet(from_scratch=config['from_scratch'])
        else:
            model = CustomCNN(norm_type=config['normalization'],
                              dropout_rate=config['dropout_rate'],
                              compression_rank=config['compression']
                              )
    num_params, model_size = get_model_size(model)
    print(f'Number of parameters: {num_params}')
    print(f'Model size: {model_size:.2f} MB')

    train_loader, test_loader = get_loaders(batch_size=config['batch_size'],
                                            resize_size=config['resize_size'],
                                            crop_size=config['crop_size']
                                            )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss() if config['loss'] == 'ce' else SquaredHingeLoss()

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, config)
    print(f'Training for: {trainer.suffix}')
    trainer.fit()
    print(f'Evaluating on test set...')
    evaluate(trainer.model, test_loader, config)
    print(f'Finished!')


for archi in ['custom', 'efficientnet_v2_s', 'resnet101']:
    print(f'Running experiments for {archi} architecture...')
    config['archi'] = archi
    if archi == 'custom':
        for normalization in ['group', 'layer', 'batch']:
            print(f'Running experiments for {normalization} normalization...')
            config['normalization'] = normalization
            experiment(config)

        for compression in [128, None]:
            print(f'Running experiments for {compression} compression...')
            config['compression'] = compression
            experiment(config)

        for dropout_rate in [0., 0.2]:
            print(f'Running experiments for {dropout_rate} dropout rate...')
            config['dropout_rate'] = dropout_rate
            experiment(config)

    else:
        for from_scratch in ['random', 'imagenet', 'tune']:
            print(f'Running experiments for {from_scratch} initialization...')
            config['from_scratch'] = from_scratch
            experiment(config)

    for weight_decay in [0., 2e-5]:
        config['weight_decay'] = weight_decay
        experiment(config)

    for loss in ['squared_hinge', 'ce']:
        config['loss'] = loss
        experiment(config)

    for learning_rate in [1e-4, 1e-2, 1e-1]:
        config['learning_rate'] = learning_rate
        experiment(config)

    for batch_size in [8, 16, 32]:
        config['batch_size'] = batch_size
        experiment(config)
