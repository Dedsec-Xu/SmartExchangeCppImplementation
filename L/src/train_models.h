#include"fpga_module.h"
#include"models.h"

void train(int epoch, int resume, float lr, float momentum, float weight_decay);
float train_one_epoch(fixed* weights, float lr, float momentum, float weight_decay);
int lenet_inference2(fixed* img, fixed* weights);
float test(fixed * weights, int print_or_not);
void cross_entropy(int size, fixed* l7, fixed* grada_l7);
float lenet_back(fixed* l0, fixed* weights, float lr, float momentum, float weight_decay);