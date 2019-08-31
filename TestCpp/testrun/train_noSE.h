#include "fpga_module.h"
#include "resnet18_model.h"
#include <iostream>
typedef float fixed;
void train(int epoch, int resume, float lr, float momentum, float weight_decay); 
float train_one_epoch(fixed weights[], float lr, float momentum, float weight_decay);
float backward(fixed* img, fixed weights[], float lr, float momentum, float weight_decay);