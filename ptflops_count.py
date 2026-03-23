# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:25:35 2023

@author: tuann
"""
from torchsummary import summary
from models.MDFNet import *
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    model = build_MDFNet(num_classes=100, width_multiplier=1.0, cifar=False, groups=1)
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True, flops_units='M',
                                             param_units='M')
    # print_per_layer_stat=True, verbose=True,param_units='M')

    # , flops_units='GMac')
    print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', macs))
    macs1 = macs.split()
    strmacs1 = str(float(macs1[0]) / 2) + ' ' + macs1[1][0]
    print('{:<30}  {:<8}'.format('Floating-point operations (FLOPs): ', strmacs1))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print('Number of model parameters (referred)): {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # summary(model, (3, 224, 224))
