import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

from torch import nn
import torch

from options import Options
from modules import SpikeLinear, SpikeConv2d, SpikeMaxPool2d


def get_model(model_name, opt, res):
    if model_name == 'feedforward':
        return FeedForward(opt, res)
    elif model_name == 'lenet5':
        return SpikingLeNet5(opt, res)
    elif model_name == 'vgg11':
        return SpikingVGG11(opt, res)
    elif model_name == 'vgg16':
        return SpikingVGG16(opt, res)
    elif model_name == 'resnet18':
        return SpikingResNet18(opt, res)
    elif model_name == 'resnet19':
        return SpikingResNet19(opt, res)
    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: feedforward, lenet5, vgg11, vgg16, resnet18, resnet19.")


class FeedForward(nn.Module):
    def __init__(self, opt, res):
        super(FeedForward, self).__init__()
        self.opt = opt
        self.res = res

        self.spike_fc1 = SpikeLinear(opt.num_inputs, opt.num_hidden, beta=opt.beta)
        self.spike_fc2 = SpikeLinear(opt.num_hidden, opt.num_outputs, beta=opt.beta)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))  # Flatten input for linear layer
        x = x.transpose(0, 1)

        spk1_rec = self.spike_fc1(x)
        spk2_rec = self.spike_fc2(spk1_rec)
        # print('spk1_rec shape:', spk1_rec.shape)
        # print('spk2_rec shape:', spk2_rec.shape)

        return spk2_rec


class SpikingLeNet5(nn.Module):
    def __init__(self, opt, res):
        super().__init__()
        in_channels = opt.in_channels
        num_classes = opt.num_classes
        beta = opt.beta

        self.features = nn.Sequential(
            SpikeConv2d(in_channels, 6, kernel_size=5, padding=2, beta=beta, use_bn=True),
            SpikeMaxPool2d(kernel_size=2, stride=2),
            SpikeConv2d(6, 16, kernel_size=5, beta=beta, use_bn=True),
            SpikeMaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            SpikeLinear(16 * 6 * 6, 120, beta=beta),
            SpikeLinear(120, 84, beta=beta),
            SpikeLinear(84, num_classes, beta=beta)
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        # print('Input shape:', x.shape)
        x = self.features(x)
        
        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)  # Flatten spatial dimensions
        
        return self.classifier(x)


class SpikingVGG11(nn.Module):
    def __init__(self, opt, res):
        super().__init__()
        in_channels = opt.in_channels
        num_classes = opt.num_classes
        beta = opt.beta

        self.features = nn.Sequential(
            SpikeConv2d(in_channels, 64, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(64, 128, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(128, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(256, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(256, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
        )
        
        # Assumes CIFAR-10 32x32 input -> pooled 5 times -> 1x1 spatial size
        self.classifier = nn.Sequential(
            SpikeLinear(512 * 1 * 1, 4096, beta=beta),
            SpikeLinear(4096, 4096, beta=beta),
            SpikeLinear(4096, num_classes, beta=beta)
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.features(x)
        # print('Shape x: ', x.shape)
        # exit()
        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)
        
        return self.classifier(x)


class SpikingVGG16(nn.Module):
    def __init__(self, opt, res, in_channels=3, num_classes=10, beta=0.9):
        super().__init__()
        in_channels = opt.in_channels
        num_classes = opt.num_classes
        beta = opt.beta
        
        # 13 Convolutional Layers
        self.features = nn.Sequential(
            SpikeConv2d(in_channels, 64, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(64, 64, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            
            SpikeConv2d(64, 128, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(128, 128, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            
            SpikeConv2d(128, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(256, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(256, 256, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            
            SpikeConv2d(256, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
            
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(512, 512, 3, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(2, 2),
        )
        
        # 3 Linear Layers
        # Note: 512 * 1 * 1 assumes CIFAR-10 32x32 input. 
        # Use AdaptiveAvgPool2d before classifier if using ImageNet.
        self.classifier = nn.Sequential(
            SpikeLinear(512 * 1 * 1, 4096, beta=beta),
            SpikeLinear(4096, 4096, beta=beta),
            SpikeLinear(4096, num_classes, beta=beta)
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.features(x)
        
        B, T, C, H, W = x.shape
        x = x.view(B, T, -1)  # Flatten spatial dimensions for linear layers
        
        return self.classifier(x)


class SpikingBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9):
        super().__init__()
        self.conv1 = SpikeConv2d(in_channels, out_channels, 3, stride=stride, padding=1, beta=beta, use_bn=True)
        self.conv2 = SpikeConv2d(out_channels, out_channels, 3, stride=1, padding=1, beta=beta, use_bn=True)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # 1x1 Conv shortcut to align dimensions
            self.shortcut = SpikeConv2d(in_channels, out_channels, 1, stride=stride, beta=beta, use_bn=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        
        # SNN Element-wise spike addition
        return out + shortcut


class SpikingResNet18(nn.Module):
    def __init__(self, opt, res):
        super().__init__()
        in_channels = opt.in_channels
        num_classes = opt.num_classes
        beta = opt.beta
        self.in_channels = 64
        
        # Stem
        self.conv1 = SpikeConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, beta=beta, use_bn=True)
        
        # Residual Layers
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1, beta=beta)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2, beta=beta)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2, beta=beta)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2, beta=beta)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpikeLinear(512, num_classes, beta=beta)

    def _make_layer(self, out_channels, num_blocks, stride, beta):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SpikingBasicBlock(self.in_channels, out_channels, s, beta))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Temporal Adaptive Pooling
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        x_pool = self.pool(x_flat)
        x_pool = x_pool.view(B, T, C)
        
        return self.fc(x_pool)


class SpikingResNet19(nn.Module):
    def __init__(self, opt, res):
        super().__init__()
        in_channels = opt.in_channels
        num_classes = opt.num_classes
        beta = opt.beta

        self.in_channels = 128 
        
        # ResNet-19 Stem: Three 3x3 convolutions replace the standard 7x7
        self.stem = nn.Sequential(
            SpikeConv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(64, 128, kernel_size=3, stride=1, padding=1, beta=beta, use_bn=True),
            SpikeConv2d(128, 128, kernel_size=3, stride=1, padding=1, beta=beta, use_bn=True),
            SpikeMaxPool2d(kernel_size=2, stride=2)
        )
        
        # Residual Layers (4 stages, 2 SEW blocks each = 16 convolutions)
        self.layer1 = self._make_layer(128, num_blocks=2, stride=1, beta=beta)
        self.layer2 = self._make_layer(256, num_blocks=2, stride=2, beta=beta)
        self.layer3 = self._make_layer(512, num_blocks=2, stride=2, beta=beta)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2, beta=beta)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SpikeLinear(512, num_classes, beta=beta)

    def _make_layer(self, out_channels, num_blocks, stride, beta):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SpikingBasicBlock(self.in_channels, out_channels, s, beta))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Temporal Adaptive Pooling
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        x_pool = self.pool(x_flat)
        x_pool = x_pool.view(B, T, C)
        
        return self.fc(x_pool)
