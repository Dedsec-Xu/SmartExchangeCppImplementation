//
//  se_conv_mask.cpp
//  
//
//  Created by LIANGDONG XU on 8/27/19.
//  Copyright © 2019 LIANGDONG XU. All rights reserved.
//
#include "fpga_module.h"
#include "SEConv.h"
#include<math.h>

//stride=1, padding=0, dilation=1, groups=1, bias=False, size_splits=64,
//threshold=5e-3 

SEConv::SEConv()
{
}

SEConv::SEConv(bool bn_input, bool relu_input, int batch_size_input, int ch_in_input,
         int ch_out_input, int size_in_input, int size_out_input, int kernel_size_input,
         int stride_input, int padding_input, fixed* conv_in_input, fixed* weights_input, fixed* bias_input,
         fixed* conv_out_input, bool bias_input, int size_splits_input, float threshold_input);
{//initialization, because C is different from python. You need to input all parameters in order to use this function,
    bn = bn_input;
    relu = relu_input;
    batch_size = batch_size_input;
    ch_in = ch_in_input;
    ch_out = ch_out_input;
    size_in = size_in_input;
    size_out = size_out_input;
    kernel_size = kernel_size_input;
    stride = stride_input;
    padding = padding_input;
    conv_in = conv_in_input;
    weights = weights_input;
    bias = bias_input;
    conv_out = conv_out_input;
    bias = bias_input;
    size_splits = size_splits_input;
    threshold = threshold_input;
    if(kernel_size > 1)
    {
        size_B = kernel_size;
        if(ch_in < size_splits)
        {
            num_splits = 1;
            size_splits = ch_in;
        }
        if(ch_in >= size_splits)
        {
            num_splits = ch_in / size_splits;
        }
    }
    else
    {

       //VEC_2_SHAPE :
        //4096: (8, 171,),
        //2048: (4, 171,),
        //1024: (2, 171,),
        //512: (1, 171,),
        //256: (1, 86,),
        //128: (1, 43,),
        //64: (1, 22,),

        size_B = 3;
        switch(ch_in)
        {
        case 4096:
            num_splits = 8;
            size_splits = 171;
            break;
        case 2048:
            num_splits = 4;
            size_splits = 171;
            break;
        case 1024:
            num_splits = 2;
            size_splits = 171;
            break;
        case 512:
            num_splits = 1;
            size_splits = 171;
            break;
        case 256:
            num_splits = 1;
            size_splits = 86;
            break;
        case 128:
            num_splits = 1;
            size_splits = 43;
            break;
        case 64:
            num_splits = 1;
            size_splits = 22;
            break;
        default :
            num_splits = 1;
            size_splits = 22;
        }
    }
	return 0;
}