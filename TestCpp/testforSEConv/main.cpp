#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "SEConv.h"
#define THRESHOLD 4e-3

using namespace std;

int main()
{

	float weights[11500200];
	const char* img_file = NULL;
    char* s1 = "test_cifar10_bin/img";
    float* img;
	char* s2 = "_";
	char* s3 = ".bin";
	//char* s4;
	std::string tmp;
    FILE *f_img;
	int num_pixels = 3072;

    tmp = std::string(s1) + std::string(s2) + "8001" + std::string(s3);
    img_file = tmp.c_str();
    f_img = fopen(img_file, "rb");
    if (!f_img) {
        std::cout << "Cannot open the file " << img_file << std::endl;
    }
    else {
        std::cout << "training using " << img_file << "............." << std::endl;
        img = new float[num_pixels * sizeof(float)];
        fread(img, sizeof(float), num_pixels + 1, f_img);
        //backward(img, weights, lr, momentum, weight_decay);
    }
    fclose(f_img);

    //float img[3072];
    float L0[65536]; //64*32*32
	float weight[1728]; //64*3*3*3
	float bias_temp64[64] = { 0.0 };
	int size_C_dim[3] = {0};
    int size_B_dim[3] = {0};
    float Ce_buffer[BF_CE_1][BF_CE_2][BF_CE_3];	//buffered Ce and B
	float B_buffer[BF_B_1][BF_B_2][BF_B_3];
    SEConv(true, true, 1, 3, 64, 32, 32, 3, 1, 2, img, weights, bias_temp64, L0, 64, 0.005, size_C_dim, size_B_dim, Ce_buffer, B_buffer);

    return 0;
}
