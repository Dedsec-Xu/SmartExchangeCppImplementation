#include <iostream>
#include <string>
#include "fpga_module.h"
// lenet for cifar-10
int lenet_inference(fixed* img, fixed* weights);
float lenet(int num_batch, int batch_size, int print_or_not);
