# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
from models.SSFNet import *

model = build_SSFNet(num_classes=1000, width_multiplier=1.0,cifar=False,groups=1)

model = model.cuda()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
#summary(model, (3, 32, 32))