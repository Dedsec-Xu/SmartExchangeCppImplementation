'''
 # @ Author: Xiaohan Chen
 # @ Email: chernxh@tamu.edu
 # @ Create Time: 2019-07-11 02:06:24
 # @ Modified by: Xiaohan Chen
 # @ Modified time: 2019-07-11 10:16:41
 # @ Description:
 '''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.utils import _pair
from quantize import quantize, quantize_grad, conv2d_biprec, QuantMeasure
from quantize import sparsify_and_nearestpow2

VEC_2_SHAPE = {
    4096: (8, 171,),
    2048: (4, 171,),
    1024: (2, 171,),
    512: (1, 171,),
    256: (1, 86,),
    128: (1, 43,),
    64: (1, 22,),
}

class SEConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding = 0, dilation = 1, groups = 1, bias = False, size_splits = 64,
                 threshold = 5e-3):
        super(SEConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.size_splits = size_splits
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.kernel_size[1] > 1:
            self.size_B = self.kernel_size[1]
            if in_channels < self.size_splits:
                self.num_splits, self.size_splits = 1, in_channels
            if in_channels >= self.size_splits:
                self.num_splits = in_channels // self.size_splits
        else:
            self.size_B = 3
            self.num_splits, self.size_splits = VEC_2_SHAPE[self.in_channels]




















            

        self.C = nn.Parameter(torch.Tensor(
                    out_channels * self.num_splits,
                    self.size_splits * self.kernel_size[0], self.size_B)).float()
        self.B = nn.Parameter(torch.Tensor(
            self.C.size()[0], self.size_B, self.size_B)).float()
        self.register_buffer('mask', torch.Tensor(*self.C.size()).float())
        self.threshold = threshold
        self.set_mask()

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_normal_(self.C, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            self.B.normal_(0, 1.0 / math.sqrt(self.size_B))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def set_mask(self):
        self.mask.data = (self.C != 0.0).float()
        assert self.mask.requires_grad == False

    def get_weight(self):
        qC = sparsify_and_nearestpow2(self.C, self.threshold)
        qC = qC * self.mask
        BC = torch.bmm(qC, self.B)
        # Reshape `self.BC` to (out_channels, in_channels, kW, hW)
        weight = BC.reshape(self.out_channels, -1, *self.kernel_size)
        if self.kernel_size[1] == 1:
            weight = weight[:,0:self.in_channels,:,:]
        return weight

    def forward(self, input):
        # Get the weight from `self.B` and `self.C`
        weight = self.get_weight()

        output = F.conv2d(input, weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        return output
