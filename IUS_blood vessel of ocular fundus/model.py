# Implementation of https://arxiv.org/pdf/1512.03385.pdf/
# See section 4.2 for model architecture on CIFAR-10.
# Some part of the code was referenced below.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1):#why not directly call class block??????????
        super(ResNet, self).__init__()
        self.in_channels = 16 
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)#False??????
        
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[0], 2)
        self.layer3 = self.make_layer(block, 128, layers[1], 2)

        self.avg_pool = nn.AvgPool2d(3,stride=2,padding=1)
        self.conv2 = conv3x3(128,64,stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3,stride=2,padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(4096, 1500)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(1500, 500)
        self.sig2 = nn.Sigmoid()
        self.fc3 = nn.Linear(500, 1)
        
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)#what's the meaning of '*'
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.max_pool(out)
        out = self.relu2(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.sig1(out)
        out = self.fc2(out)
        out = self.sig2(out)
        out = self.fc3(out)
        #print out.shape
        return nn.Sigmoid()(out)#why dose the 'out' is outside??????????
