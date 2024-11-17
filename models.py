"""
Author: Dimas Ahmad
Description: This file contains the model classes for the project.
"""

import torch
import torchvision


def get_efficient_net(num_classes=10, from_scratch='tune'):
    weight = None if from_scratch == 'random' else 'DEFAULT'
    model = torchvision.models.efficientnet_v2_s(weights=weight)

    if from_scratch == 'tune':
        for param in model.parameters():
            param.requires_grad = False

    last_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(last_features, num_classes)
    return model


def get_resnet(num_classes=10, from_scratch='tune'):
    weight = None if from_scratch == 'random' else 'DEFAULT'
    model = torchvision.models.resnet101(pretrained=weight)

    if from_scratch == 'tune':
        for param in model.parameters():
            param.requires_grad = False

    last_features = model.fc.in_features
    model.fc = torch.nn.Linear(last_features, num_classes)
    return model


class CustomCNN(torch.nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=10,
                 norm_type='batch',
                 dropout_rate=0.2,
                 num_groups=8,
                 compression_rank=None
                 ):
        """
        Custom CNN with flexible normalization and optional FC compression.

        Architecture:
        - 3 convolutional blocks (conv -> norm -> relu -> pool)
        - 2 FC layers with optional compression

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            norm_type: Type of normalization to use
            num_groups: Number of groups for GroupNorm
            dropout_rate: Dropout rate
            compression_rank: If set, compress FC layer using Truncated SVD
        """
        super().__init__()

        def get_norm_layer(channels, norm_type):
            if norm_type == 'batch':
                return torch.nn.BatchNorm2d(channels)
            elif norm_type == 'layer':
                return torch.nn.LayerNorm(channels)
            elif norm_type == 'group':
                return torch.nn.GroupNorm(num_groups, channels)
            else:
                raise ValueError(f'Normalization type {norm_type} not supported.')

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            get_norm_layer(64, norm_type),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.1),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            get_norm_layer(128, norm_type),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.1),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            get_norm_layer(256, norm_type),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.1)
        )

        feature_size = 256*(224//(2**3))*(224//(2**3))

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(512, num_classes)
        )

        self.fc_compressed_rank = compression_rank
        if compression_rank is not None:
            self.compress_fc()

    def compress_fc(self):
        """
        Compress FC layer using Truncated SVD
        """
        fc_layer = None
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):
                fc_layer = layer
                break

        if fc_layer is None:
            return

        U, S, V = torch.linalg.svd(fc_layer.weight.data, full_matrices=False)
        rank = min(self.compression_rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        V = V[:rank, :]

        compressed_fc1 = torch.nn.Linear(V.size(1), rank, bias=False)
        compressed_fc2 = torch.nn.Linear(rank, U.size(0), bias=fc_layer.bias is not None)

        with torch.no_grad():
            compressed_fc1.weight.copy_(V)
            compressed_fc2.weight.copy_(U*S[:, None])
            if fc_layer.bias is not None:
                compressed_fc2.bias.copy_(fc_layer.bias)

        compressed_classifier = torch.nn.Sequential()
        replace = False
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear) and not replace:
                compressed_classifier.append(compressed_fc1)
                compressed_classifier.append(torch.nn.ReLU(inplace=True))
                compressed_classifier.append(compressed_fc2)
                replace = True
            else:
                compressed_classifier.append(layer)

        self.classifier = compressed_classifier

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
