#ifndef SE_CONV_MASK_H
#define SE_CONV_MASK_H
#include "fpga_module.h"

class SEConv
{       
public:
    SEConv();
    SEConv(bool bn_input, bool relu_input, int batch_size_input, int ch_in_input,
         int ch_out_input, int size_in_input, int size_out_input, int kernel_size_input,
         int stride_input, int padding_input, fixed* conv_in_input, fixed* weights_input, fixed* bias_input,
         fixed* conv_out_input, bool bias_input, int size_splits_input, float threshold_input);
    bool bn;
    bool relu;
    int batch_size;
    int ch_in;
    int ch_out;
    int size_in;
    int size_out;
    int kernel_size;
    int stride;
    int padding;
    fixed* conv_in;
    fixed* weights;
    fixed* bias;
    fixed* conv_out;
    bool bias;
    int size_splits;
    float threshold;
    int size_B;
    int size_C;
    int num_splits;
}





#endif