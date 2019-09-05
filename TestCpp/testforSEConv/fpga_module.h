#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include <ctime>
#include "stdio.h"
#include <cmath>
typedef float fixed;

int conv(bool bn, bool relu, int batch_size, int ch_in, int ch_out, int size_in, int size_out, int kernel_size, int stride, int padding, fixed* conv_in, fixed* weights, fixed* bias, fixed* conv_out);

int fc(bool bn, bool bias_or_not, bool relu, int batch_size, int ch_in, int ch_out, fixed* fc_in, fixed* weights, fixed* bias, fixed* fc_out);

int max_pooling(int batch_size, int ch, int size_in, int size_out, int kernel_size, fixed* pooling_in, fixed* pooling_out);

int get_label(int num_classes, fixed* vec_in);

void batch_norm(int batch_size, int channel, int size, fixed* input);

int max_pooling2(int batch_size, int ch, int size_in, int size_out, int kernel_size, fixed* pooling_in, fixed* pooling_out, int* max_position);

void ave_pooling4x4(int ch,int size_in,int size_out,fixed pooling_in[], fixed pooling_out[]);

void matrix_add(int size, fixed m1[], fixed m2[],fixed sum[]);

void softmax_back(int label, int size, fixed* out_layer, fixed* grada);

void conv_back(fixed gradw[], fixed gradb[], fixed grada[], fixed grada_last[], fixed a[], fixed a_last[], fixed grada_part[], fixed gradw_part[], fixed grada_last_part[], fixed a_part[], fixed w_part[], fixed weights[]);

void fc_back(fixed gradw[], fixed gradb[], fixed grada[], fixed grada_last[], fixed a[], fixed weights[], fixed bias[]);