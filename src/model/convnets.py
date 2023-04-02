import torch as pt
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable

class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Down1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Up1D, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class PixelShuffle1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels * upscale_factor, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels * upscale_factor)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pixel_shuffle(out)
        return out

class PixelUnshuffle1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels // downscale_factor, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels // downscale_factor)
        self.relu = nn.ReLU()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pixel_unshuffle(out)
        return out



