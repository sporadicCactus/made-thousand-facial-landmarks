import torch
from torch import nn
from torch.nn import functional as F

import random

import torchvision
from torchvision.models import ResNet



from utils import NUM_PTS

try:
    from mish_cuda import MishCuda as Mish
except:
    class Mish(nn.Module):
        def forward(self, x):
            return x*torch.tanh(F.softplus(x))


def mod_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    w = x
    x = self.layer2(x)
    z = x
    x = self.layer3(x)
    y = x
    x = self.layer4(x)
    return w, z, y, x

ResNet.forward = mod_forward

def replace_relu_with_mish(module):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if isinstance(target_attr, nn.ReLU):
            setattr(module, attr_str, Mish())
        try:
            for ch in target_attr.children():
                replace_relu_with_mish(ch)
        except:
            pass

class CAMaxPool2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.Identity()):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv_1x1 = nn.Conv2d(3*channels, channels, 1)
        self.conv_1x1.weight.data = torch.cat([torch.diag(torch.ones(channels))[...,None,None],
                                               torch.zeros(channels, channels*2, 1, 1)], dim=1)
        self.conv_1x1.bias.data = torch.zeros(channels)
        self.activation = activation


    def forward(self, x):
        type_ = x.type()
        device_ = x.device

        res, ind = F.max_pool2d_with_indices(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)

        ind = ind % (x.shape[-1]*x.shape[-2])
        w = ind %  x.shape[-1] - torch.arange(self.kernel_size//2 - 1, x.shape[-1], self.stride, device=device_).type(type_)
        h = ind // x.shape[-1] - torch.arange(self.kernel_size//2 - 1, x.shape[-2], self.stride, device=device_)[:,None].type(type_)

        res = torch.cat([res, h, w], dim=-3)
        res = self.conv_1x1(res)
        res = self.activation(res)

        return res

class GlobalCAMaxPool2d(nn.Module):
    def forward(self, x):
        type_ = x.type()
        device_ = x.device

        res, ind = F.max_pool2d_with_indices(x, kernel_size=x.shape[-2:])

        ind = ind % (x.shape[-1]*x.shape[-2])
        w = ind %  x.shape[-1]
        w = (w.type(type_) + 0.5)/x.shape[-2] - 0.5
        h = ind // x.shape[-1]
        h = (h.type(type_) + 0.5)/x.shape[-1] - 0.5

        res = torch.cat([res, h, w], dim=-3)[...,0,0]
        return res

def mod_resnet50():
    model = torchvision.models.resnet50(pretrained=True)
    model.maxpool = CAMaxPool2d(channels=64, kernel_size=3, stride=2, padding=1, activation=Mish())
    del model.avgpool
    del model.fc
    replace_relu_with_mish(model)
    return model

class ModResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mod_resnet50()
        self.head = nn.Sequential(
                                GlobalCAMaxPool2d(),
                                nn.Linear(2048*3, 2048),
                                Mish(),
                                nn.Linear(2048, 2*NUM_PTS),   
                                 ) 
        self.fpn_mode = False

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]
        x = self.head(x)
        x = x.reshape(-1, NUM_PTS, 2)
        if self.fpn_mode:
            return x, features
        else:
            return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            Mish()
        )

    def forward(self, x):
        return self.conv(x)
        
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
        )

    def forward(self, x, target_size):
        x = self.upsample(x)
        x = F.interpolate(x, target_size, mode='bilinear', align_corners=False)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)

class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()

        self.initial_transforms = nn.ModuleList([Conv(channels, channels//2, 1) for channels in feature_channels[:-1]] +\
                                                [nn.Sequential(
                                                    Conv(feature_channels[-1], feature_channels[-1]//2, 1),
                                                    Conv(feature_channels[-1]//2, feature_channels[-1], 3),
                                                    Conv(feature_channels[-1], feature_channels[-1]//2, 1))
                                                ])

        self.downstream_transforms = nn.ModuleList([self._horizontal_stack(channels) for channels in feature_channels[:-1]] + [nn.Identity()])
        self.upstream_transforms = nn.ModuleList([nn.Identity()] + [self._horizontal_stack(channels) for channels in feature_channels[1:]])

        self.upsamplers = nn.ModuleList([Upsample(channel_high//2, channel_low//2) for channel_high, channel_low in zip(feature_channels[1:], feature_channels[:-1])])
        self.downsamplers = nn.ModuleList([Downsample(channel_low//2, channel_high//2) for channel_low, channel_high in zip(feature_channels[:-1], feature_channels[1:])])

    def _horizontal_stack(self, channels):
        return nn.Sequential(
            Conv(channels, channels//2, 1),
            Conv(channels//2, channels, 3),
            Conv(channels, channels//2, 1),
            Conv(channels//2, channels, 3),
            Conv(channels, channels//2, 1),
        )

    def forward(self, features):
        features = [tr(f) for tr, f in zip(self.initial_transforms, features)]

        # descending the feature pyramid
        features[-1] = self.downstream_transforms[-1](features[-1])
        for ind in range(len(features) - 1, 0, -1):
            features[ind - 1] = torch.cat([features[ind - 1], self.upsamplers[ind - 1](features[ind], features[ind - 1].shape[-2:])], dim=1)
            features[ind - 1] = self.downstream_transforms[ind - 1](features[ind - 1])


        # ascending the feature pyramid
        features[0] = self.upstream_transforms[0](features[0])
        for ind in range(0, len(features) - 1, +1):
            features[ind + 1] = torch.cat([self.downsamplers[ind](features[ind]), features[ind + 1]], dim=1)
            features[ind + 1] = self.upstream_transforms[ind + 1](features[ind + 1])

        return tuple(features)

class Resnet50_PANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mod_resnet50()
        self.panet = PANet([512, 1024, 2048])
        self.ca_maxpool = GlobalCAMaxPool2d()
        self.head =  nn.Sequential( 
                                nn.Linear((1024 + 512 + 256)*3, 4096),
                                Mish(),
                                nn.Linear(4096, 2*NUM_PTS),   
                                 )

    def forward(self, x):
        x = self.backbone(x)[-3:]
        features = self.panet(x)
        x = torch.cat([self.ca_maxpool(f) for f in features], dim=1)
        x = self.head(x)
        x = x.reshape(-1, NUM_PTS, 2)
        return x
