#include "fpga_module.h"
#include"train_models.h"
#include"unistd.h"



int resnet18_inference(fixed* img, fixed* weights);
float resnet18(int num_batch, int batch_size,int num_weights, int print_or_not);



