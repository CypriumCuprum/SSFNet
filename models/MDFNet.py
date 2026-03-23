import re
import types

import torch.nn
import torch.nn.init
from .SE_Attention import *
import torch.nn.functional as F
from .common import conv1x1_block, conv3x3_block, conv5x5_dw_dilation_block, conv3x3_dw_dilation_block, Classifier,conv1x1_group_block, conv7x7_dw_dilation_block

class MDF(torch.nn.Module):  # defined for multiple dilation features (MDF)
    """
    Based on multiple depthwise dilation features(MDDF)
    MDFNet: Congregating multiple dilation features for image classification
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride, groups=1):
        super().__init__()
        self.groups = groups
        self.Pw1 = conv1x1_block(in_channels=in_channels,
                        out_channels=in_channels,                                
                        use_bn=False,
                        activation=None)
        self.dw1 = conv3x3_dw_dilation_block(channels=in_channels, stride=stride,dilation=1,activation=None)
        self.dw2 = conv3x3_dw_dilation_block(channels=in_channels, stride=stride,dilation=2)
        self.dw3 = conv5x5_dw_dilation_block(channels=in_channels, stride=stride,dilation=3)
        self.dw4 = conv7x7_dw_dilation_block(channels=in_channels, stride=stride,dilation=4)
        
        self.Pw2 = conv1x1_group_block(in_channels=in_channels,
                                        out_channels=out_channels,groups=groups)
        
        if stride == 2 or in_channels != out_channels: 
            self.PwRes = conv1x1_block(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SE = SE(out_channels, 16)

    def forward(self, x):
        residual = x
        x = self.Pw1(x)
        x_dilated1 = self.dw1(x)
        batch_size,num_channel,h,w = x_dilated1.size()
        x_dilated2 = self.dw2(x)
        x_dilated2 = F.interpolate(x_dilated2,size=h,mode='nearest')
        x_dilated3 = self.dw3(x)
        x_dilated3 = F.interpolate(x_dilated3,size=h,mode='nearest')
        x_dilated4 = self.dw4(x)
        x_dilated4 = F.interpolate(x_dilated4,size=h,mode='nearest')
        
        x = F.relu(x_dilated1 * F.sigmoid(x_dilated2 + x_dilated3 + x_dilated4))
        
        x = self.Pw2(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else: 
            residual = self.PwRes(residual)
            x = x + residual
        return x


class MDFNet(torch.nn.Module):
    """
    """

    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True, groups=2):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module("unit{}".format(unit_id + 1),
                                 MDF(in_channels=in_channels, out_channels=unit_channels, stride=stride, groups=groups))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 1024
        self.backbone.add_module("final_conv",
                                 conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels,
                                               activation="relu"))
        self.backbone.add_module("dropout1", torch.nn.Dropout2d(0.2))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.backbone.add_module("dropout2", torch.nn.Dropout2d(0.2))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def build_MDFNet(num_classes, width_multiplier=1.0, cifar=False, groups=2):
    init_conv_channels = 32
    channels = [[64], [64, 128], [128, 256,256], [256, 512, 512], [512]]

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
        # strides = [1, 1, 1, 2, 2]#strideNew
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    if width_multiplier != 1.0:
        channels = [[int(unit * width_multiplier) for unit in stage] for stage in channels]
        init_conv_channels = int(init_conv_channels * width_multiplier)

    return MDFNet(num_classes=num_classes,
                  init_conv_channels=init_conv_channels,
                  init_conv_stride=init_conv_stride,
                  channels=channels,
                  strides=strides,
                  in_size=in_size, groups=groups)
