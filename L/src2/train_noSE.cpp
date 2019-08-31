#include "fpga_module.h"
#include "resnet18_model.h"
#include "train_noSE.h"
#include <random>

void train(int epoch, int resume, float lr, float momentum, float weight_decay) {
	const int num_weights = 11500200;
	int start_epoch;
	float best_acc;
	float acc = 0;
	float weights[11500200];
	if (resume == 0) {
		std::default_random_engine generator;
		std::normal_distribution<float> distribution(0.0, 1.0);
		for (int i = 0; i < num_weights; i++) {
			weights[i] = distribution(generator);
		}
		start_epoch = 0;
	}
	else {
		 FILE * f_we = fopen("trained_weights.bin","rb");
		 FILE * f_epoch = fopen("epoch.bin","rb");
		 FILE * f_acc = fopen("best_acc.bin","rb");
		 fread(weights, sizeof(float), 11672232,f_we);
		 fread(&start_epoch, sizeof(int), 1, f_epoch);
		 fread(&best_acc, sizeof(float), 1, f_acc);
		 fclose(f_acc);
		 fclose(f_epoch);
		 fclose(f_we);
	}

	int print = 0;
	FILE * f_we_w;
	FILE * f_epoch_w;
	FILE * f_acc_w;
	for (int i = start_epoch; i < start_epoch + epoch; i++) {
		std::cout << "epoch: " << i << " training............" << std::endl;
		train_one_epoch(weights, lr, momentum, weight_decay);
		std::cout << "epoch: " << i << " testing............" << std::endl;
		//acc = test(weights, print);
		//std::cout << "weights:" << std::endl;
		//for (int w = 0; w < num_weights; w++) {
		//	std::cout << weights[w] << " ";
		//}
		//std::cout << std::endl;
		std::cout << "Test acc is : " << acc << std::endl;
	}
}

float train_one_epoch(fixed weights[], float lr, float momentum, float weight_decay) {
	const char* img_file = NULL;
	char* s1 = "test_cifar10_bin/img";
	char* s2 = "_";
	char* s3 = ".bin";
	std::string tmp;
	FILE *f_img;
	int num_pixels = 3072;
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 2; j += 1) {
			tmp = std::string(s1) + std::string(s2) + std::to_string(i * 100 + j) + std::string(s3);
			img_file = tmp.c_str();
			f_img = fopen(img_file, "rb");
			if (!f_img) {
				std::cout << "Cannot open the file " << img_file << std::endl;
			}
			else {
				std::cout << "procssing " << img_file << "............." << std::endl;
				fixed* img = new fixed[num_pixels * sizeof(fixed)];
				fread(img, sizeof(fixed), num_pixels + 1, f_img);
				backward(img, weights, lr, momentum, weight_decay);
			}
			fclose(f_img);
		}
	}
	return 0;
}

float backward(fixed* img, fixed weights[], float lr, float momentum, float weight_decay) {
	int weight_id;
	int bias_id;
	int num_pixels = 3072;
	int label = img[num_pixels];

	//layer0**************************************************
	fixed L0[65536]; //64*32*32
	weight_id = 0; //64*3*3*3
	fixed bias_temp64[64] = { 0.0 };
	conv(true, true, 1, 3, 64, 32, 32, 3, 1, 2, img, weights + weight_id, bias_temp64, L0);
	delete[] img;
	img = NULL;

	//layer1**************************************************
	//block1
	fixed L11[65536];//64*32*32
	weight_id = weight_id + 1728;//64*64*3*3
	conv(true, true, 1, 64, 64, 32, 32, 3, 1, 2, L0, weights + weight_id, bias_temp64, L11);

	fixed L12[65536];//64*32*32
	weight_id = weight_id + 36864;//64*64*3*3
	conv(true, false, 1, 64, 64, 32, 32, 3, 1, 2, L11, weights + weight_id, bias_temp64, L12);

	fixed L12_[65536];//64*32*32
	matrix_add(65536, L12, L0, L12_); //Residual+ReLU

	//block2
	fixed L13[65536];//64*32*32
	weight_id = weight_id + 36864;//64*3*3*3
	conv(true, true, 1, 64, 64, 32, 32, 3, 1, 2, L12_, weights + weight_id, bias_temp64, L13);

	fixed L14[65536];//64*32*32
	weight_id = weight_id + 36864;//64*3*3*3
	conv(true, false, 1, 64, 64, 32, 32, 3, 1, 2, L13, weights + weight_id, bias_temp64, L14);

	fixed L14_[65536];//64*32*32
	matrix_add(65536, L14, L12_, L14_); //Residual+ReLU

	//layer2**************************************************
	//block1
	fixed L21[32768];//128*16*16
	weight_id = weight_id + 36864;//128*64*3*3
	fixed bias_temp128[128] = { 0.0 };
	conv(true, true, 1, 64, 128, 32, 16, 3, 2, 2, L14_, weights + weight_id, bias_temp128, L21);

	fixed L22[32768];//128*16*16
	weight_id = weight_id + 73728;//128*128*3*3
	conv(true, false, 1, 128, 128, 16, 16, 3, 1, 2, L21, weights + weight_id, bias_temp128, L22);

	fixed L14_down[32768];//128*16*16
	fixed weight_down_L1[8192];//128*64*1*1
	for (int i = 0; i < 8192; i++) {
		weight_down_L1[i] = 1;
	}
	conv(true, false, 1, 64, 128, 32, 16, 1, 2, 0, L14_, weight_down_L1, bias_temp128, L14_down);//down sample
	fixed L22_[32768];//128*16*16
	matrix_add(32768, L22, L14_down, L22_); //Residual+ReLU 

	//block2
	fixed L23[32768];//128*16*16
	weight_id = weight_id + 147456;//128*128*3*3
	conv(true, true, 1, 128, 128, 16, 16, 3, 1, 2, L22_, weights + weight_id, bias_temp128, L23);

	fixed L24[32768];//128*16*16
	weight_id = weight_id + 147456;//128*128*3*3
	conv(true, false, 1, 128, 128, 16, 16, 3, 1, 2, L23, weights + weight_id, bias_temp128, L24);

	fixed L24_[32768];//128*16*16
	matrix_add(32768, L24, L22_, L24_);//Residual+ReLU

	//layer3**************************************************
	//block1
	fixed L31[16384];//256*8*8
	weight_id = weight_id + 147456;//256*128*3*3
	fixed bias_temp256[256] = { 0.0 };
	conv(true, true, 1, 128, 256, 16, 8, 3, 2, 2, L24_, weights + weight_id, bias_temp256, L31);

	fixed L32[16384];//256*8*8
	weight_id = weight_id + 294912;//256*256*3*3
	conv(true, false, 1, 256, 256, 16, 8, 3, 1, 2, L31, weights + weight_id, bias_temp256, L32);

	fixed L24_down[16384];
	fixed weight_down_L2[32768];//256*128*1*1
	for (int i = 0; i < 32768; i++) {
		weight_down_L2[i] = 1;
	}
	conv(true, false, 1, 128, 256, 16, 8, 1, 2, 0, L24_, weight_down_L2, bias_temp256, L24_down);//down sample
	fixed L32_[16384];//256*8*8
	matrix_add(16384, L32, L24_down, L32_); //Residual+ReLU 

	//block2
	fixed L33[16384];//256*8*8
	weight_id = weight_id + 589824;//256*256*3*3
	conv(true, true, 1, 256, 256, 16, 8, 3, 1, 2, L32_, weights + weight_id, bias_temp256, L33);

	fixed L34[16384];//256*8*8
	weight_id = weight_id + 589824;//256*256*3*3
	conv(true, false, 1, 256, 256, 16, 8, 3, 1, 2, L33, weights + weight_id, bias_temp256, L34);

	fixed L34_[16384];//256*8*8
	matrix_add(16384, L34, L32_, L34_); //Residual+ReLU 

	//layer4**************************************************
	fixed L41[8192];//512*4*4
	weight_id = weight_id + 589824;//512*256*3*3
	fixed bias_temp512[512] = { 0.0 };
	conv(true, true, 1, 256, 512, 8, 4, 3, 2, 2, L34_, weights + weight_id, bias_temp512, L41);

	fixed L42[8192];//512*4*4
	weight_id = weight_id + 1179648;//512*512*3*3
	conv(true, false, 1, 512, 512, 8, 4, 3, 1, 2, L41, weights + weight_id, bias_temp512, L42);

	fixed L34_down[8192];//512*4*4
	fixed weight_down_L3[131072];//512*256*1*1
	for (int i = 0; i < 131072; i++) {
		weight_down_L3[i] = 1;
	}
	conv(true, false, 1, 256, 512, 8, 4, 1, 2, 0, L34_, weight_down_L3, bias_temp512, L34_down);//down sample
	fixed L42_[8192];//512*4*4
	matrix_add(8192, L42, L34_down, L42_); //Residual+ReLU 

	//block2
	fixed L43[8192];//512*4*4
	weight_id = weight_id + 2359296;//512*512*3*3
	conv(true, true, 1, 512, 512, 8, 4, 3, 1, 2, L42_, weights + weight_id, bias_temp512, L43);

	fixed L44[8192];//512*4*4
	weight_id = weight_id + 2359296;//512*512*3*3
	conv(true, true, 1, 512, 512, 8, 4, 3, 1, 2, L43, weights + weight_id, bias_temp512, L44);

	fixed L44_[8192];//512*4*4
	matrix_add(8192, L44, L42, L44_); //Residual+ReLU 

	//layer5 Avepooling*****************************************************
	fixed L5[512];//512*1*1;
	ave_pooling4x4(512, 4, 1, L44_, L5);

	//layer6 Fully connected************************************************
	fixed L6[1000];
	weight_id = weight_id + 2359296;
	bias_id = weight_id + 512000;
	fixed bias_temp1000[1000] = { 0.0 };
	fc(false, true, false, 1, 512, 1000, L5, weights + weight_id, weights + bias_id, L6);


	//Backward******************************************************************************************

	//L6
	fixed grada_6[1000];
	softmax_back(label, 1000,L6, grada_6);
	//L5
	fixed gradw_5[512000];
	fixed gradb_5[1000];
	fixed grada_5[512];
	fc_back(gradw_5, gradb_5, grada_5, grada_6, L5, weights + weight_id, weights+bias_id);
	for (int i = 0; i < 1000; i++) {
		weights[bias_id + i] += -lr * gradb_5[i];
		for (int j = 0; j < 512; j++) {
			weights[weight_id + i * 512 + j] += -lr * gradw_5[i * 512 + j];
		}
	}

	//L44
    fixed grada_44[8192];//512*4*4               
	for (int i = 0; i < 512; i++) {
		for (int j = 0; j < 4; j++) {
			for (int k = 0; k < 4; j++) {
				grada_44[i * 16 + j * 4 + k] = grada_5[i] / 16;
			}
		}
	}

	//L43
	fixed grada_43[8192];//512*4*4
	fixed gradw_43[2359296];//512*512*3*3
	fixed gradb_43[512] = { 0 };//no bias
	fixed grada_43_part[16];//4*4
	fixed gradw_43_part[9];//3*3
	fixed grada_44_part[16];//4*4
	fixed a_43_part[16];//4*4
	fixed w_43_part[9];//3*3
	weight_id = weight_id - 2359296;
	conv_back(gradw_43, gradb_43, grada_43, grada_44, L43, L44, grada_43_part, gradw_43_part, grada_44_part, a_43_part, w_43_part, weights + weight_id);
	for (int i = 0; i < 2359296; i++) {
		weights[weight_id + i] += -lr * gradw_43[i];
	}

	//L42
	fixed grada_42[8192];//512*4*4               
	fixed gradw_42[2359296];//512*512*3*3
	fixed gradb_42[512] = { 0 };//no bias
	fixed grada_42_part[16];//4*4
	fixed gradw_42_part[9];//3*3
	//fixed grada_43_part[16];//4*4
	fixed a_42_part[16];//4*4
	fixed w_42_part[9];//3*3
	weight_id = weight_id - 2359296;
	conv_back(gradw_42, gradb_42, grada_42, grada_43, L42, L43, grada_42_part, gradw_42_part, grada_43_part, a_43_part, w_42_part, weights + weight_id);
	for (int i = 0; i < 2359296; i++) {
		weights[weight_id + i] += -lr * (gradw_42[i]+1);
	}

	//L41
	fixed grada_41[8192];//512*4*4
	fixed gradw_41[2359296];//512*512*3*3
	fixed gradb_41[512] = { 0 };//no bias
	fixed grada_41_part[16];//4*4
	fixed gradw_41_part[9];//3*3
	//fixed grada_42_part[16];//4*4
	fixed a_41_part[16];//4*4
	fixed w_41_part[9];//3*3
	weight_id = weight_id - 2359296;
	conv_back(gradw_41, gradb_41, grada_41, grada_42, L41, L42, grada_41_part, gradw_41_part, grada_42_part, a_41_part, w_41_part, weights + weight_id);
	for (int i = 0; i < 2359296; i++) {
		weights[weight_id + i] += -lr * gradw_41[i];
	}

	//L34
	fixed grada_34[16384];//256*8*8
	fixed gradw_34[1179648];//512*256*3*3
	fixed gradb_34[512] = { 0 };//no bias
	fixed grada_34_part[64];//8*8
	fixed gradw_34_part[9];//3*3
	//fixed grada_41_part[16];//4*4
	fixed a_34_part[64];//8*8
	fixed w_34_part[9];//3*3
	weight_id = weight_id - 1179648;
	conv_back(gradw_34, gradb_34, grada_34, grada_41, L34, L41, grada_34_part, gradw_34_part, grada_41_part, a_34_part, w_34_part, weights + weight_id);
	for (int i = 0; i < 1179648; i++) {
		weights[weight_id + i] += -lr * (gradw_34[i] + 1);
	}

    //L33
	fixed grada_33[16384];//256*8*8
	fixed gradw_33[589824];//256*256*3*3
	fixed gradb_33[256] = { 0 };//no bias
	fixed grada_33_part[64];//8*8
	fixed gradw_33_part[9];//3*3
	//fixed grada_34_part[64];//8*8
	fixed a_33_part[64];//8*8
	fixed w_33_part[9];//3*3
	weight_id = weight_id - 589824;
	conv_back(gradw_33, gradb_33, grada_33, grada_34, L33, L34, grada_33_part, gradw_33_part, grada_34_part, a_33_part, w_33_part, weights + weight_id);
	for (int i = 0; i < 589824; i++) {
		weights[weight_id + i] += -lr * (gradw_33[i]);
	}

	//L32
	fixed grada_32[16384];//256*8*8
	fixed gradw_32[589824];//256*256*3*3
	fixed gradb_32[256] = { 0 };//no bias
	fixed grada_32_part[64];//8*8
	fixed gradw_32_part[9];//3*3
	//fixed grada_33_part[64];//8*8
	fixed a_32_part[64];//8*8
	fixed w_32_part[9];//3*3
	weight_id = weight_id - 589824;
	conv_back(gradw_32, gradb_32, grada_32, grada_33, L32, L33, grada_32_part, gradw_32_part, grada_33_part, a_32_part, w_32_part, weights + weight_id);
	for (int i = 0; i < 589824; i++) {
		weights[weight_id + i] += -lr * (gradw_32[i]+1);
	}

	//L31
	fixed grada_31[16384];//256*8*8
	fixed gradw_31[589824];//256*256*3*3
	fixed gradb_31[256] = { 0 };//no bias
	fixed grada_31_part[64];//8*8
	fixed gradw_31_part[9];//3*3
	//fixed grada_32_part[64];//8*8
	fixed a_31_part[64];//8*8
	fixed w_31_part[9];//3*3
	weight_id = weight_id - 589824;
	conv_back(gradw_31, gradb_31, grada_31, grada_32, L31, L32, grada_31_part, gradw_31_part, grada_32_part, a_31_part, w_31_part, weights + weight_id);
	for (int i = 0; i < 589824; i++) {
		weights[weight_id + i] += -lr * (gradw_31[i]);
	}

    //L24
	fixed grada_24[32768];//128*16*16
	fixed gradw_24[294912];//256*128*3*3
	fixed gradb_24[256] = { 0 };//no bias
	fixed grada_24_part[256];//16*16
	fixed gradw_24_part[9];//3*3
	//fixed grada_31_part[64];//8*8
	fixed a_24_part[256];//16*16
	fixed w_24_part[9];//3*3
	weight_id = weight_id - 294912;
	conv_back(gradw_24, gradb_24, grada_24, grada_31, L24, L31, grada_24_part, gradw_24_part, grada_31_part, a_24_part, w_24_part, weights + weight_id);
	for (int i = 0; i < 294912; i++) {
		weights[weight_id + i] += -lr * (gradw_24[i]+1);
	}

	//L23
	fixed grada_23[32768];//128*16*16
	fixed gradw_23[147456];//128*128*3*3
	fixed gradb_23[128] = { 0 };//no bias
	fixed grada_23_part[256];//16*16
	fixed gradw_23_part[9];//3*3
	//fixed grada_24_part[256];//16*16
	fixed a_23_part[256];//16*16
	fixed w_23_part[9];//3*3
	weight_id = weight_id - 147456;
	conv_back(gradw_23, gradb_23, grada_23, grada_24, L23, L24, grada_23_part, gradw_23_part, grada_24_part, a_23_part, w_23_part, weights + weight_id);
	for (int i = 0; i < 147456; i++) {
		weights[weight_id + i] += -lr * (gradw_23[i]);
	}

	//L22
	fixed grada_22[32768];//128*16*16
	fixed gradw_22[147456];//128*128*3*3
	fixed gradb_22[128] = { 0 };//no bias
	fixed grada_22_part[256];//16*16
	fixed gradw_22_part[9];//3*3
	//fixed grada_23_part[256];//16*16
	fixed a_22_part[256];//16*16
	fixed w_22_part[9];//3*3
	weight_id = weight_id - 147456;
	conv_back(gradw_22, gradb_22, grada_22, grada_23, L22, L23, grada_22_part, gradw_22_part, grada_23_part, a_22_part, w_22_part, weights + weight_id);
	for (int i = 0; i < 147456; i++) {
		weights[weight_id + i] += -lr * (gradw_22[i]+1);
	}

	//L21
	fixed grada_21[32768];//128*16*16
	fixed gradw_21[147456];//128*128*3*3
	fixed gradb_21[128] = { 0 };//no bias
	fixed grada_21_part[256];//16*16
	fixed gradw_21_part[9];//3*3
	//fixed grada_22_part[256];//16*16
	fixed a_21_part[256];//16*16
	fixed w_21_part[9];//3*3
	weight_id = weight_id - 147456;
	conv_back(gradw_21, gradb_21, grada_21, grada_22, L21, L22, grada_21_part, gradw_21_part, grada_22_part, a_21_part, w_21_part, weights + weight_id);
	for (int i = 0; i < 147456; i++) {
		weights[weight_id + i] += -lr * (gradw_21[i]);
	}

	//L14
	fixed grada_14[65536];//64*32*32
	fixed gradw_14[73728];//128*64*3*3
	fixed gradb_14[128] = { 0 };//no bias
	fixed grada_14_part[1024];//32*32
	fixed gradw_14_part[9];//3*3
	//fixed grada_21_part[256];//16*16
	fixed a_14_part[1024];//32*32
	fixed w_14_part[9];//3*3
	weight_id = weight_id - 73728;
	conv_back(gradw_14, gradb_14, grada_14, grada_21, L14, L21, grada_14_part, gradw_14_part, grada_21_part, a_14_part, w_14_part, weights + weight_id);
	for (int i = 0; i < 73728; i++) {
		weights[weight_id + i] += -lr * (gradw_14[i]+1);
	}

	//L13
	fixed grada_13[65536];//64*32*32
	fixed gradw_13[36864];//64*64*3*3
	fixed gradb_13[64] = { 0 };//no bias
	fixed grada_13_part[1024];//32*32
	fixed gradw_13_part[9];//3*3
	//fixed grada_14_part[1024];//32*32
	fixed a_13_part[1024];//32*32
	fixed w_13_part[9];//3*3
	weight_id = weight_id - 36864;
	conv_back(gradw_13, gradb_13, grada_13, grada_14, L13, L14, grada_13_part, gradw_13_part, grada_14_part, a_13_part, w_13_part, weights + weight_id);
	for (int i = 0; i < 36864; i++) {
		weights[weight_id + i] += -lr * (gradw_13[i]);
	}

	//L12
	fixed grada_12[65536];//64*32*32
	fixed gradw_12[36864];//64*64*3*3
	fixed gradb_12[64] = { 0 };//no bias
	fixed grada_12_part[1024];//32*32
	fixed gradw_12_part[9];//3*3
	//fixed grada_13_part[1024];//32*32
	fixed a_12_part[1024];//32*32
	fixed w_12_part[9];//3*3
	weight_id = weight_id - 36864;
	conv_back(gradw_12, gradb_12, grada_12, grada_13, L12, L13, grada_12_part, gradw_12_part, grada_13_part, a_12_part, w_12_part, weights + weight_id);
	for (int i = 0; i < 36864; i++) {
		weights[weight_id + i] += -lr * (gradw_12[i]+1);
	}

	//L11
	fixed grada_11[65536];//64*32*32
	fixed gradw_11[36864];//64*64*3*3
	fixed gradb_11[64] = { 0 };//no bias
	fixed grada_11_part[1024];//32*32
	fixed gradw_11_part[9];//3*3
	//fixed grada_12_part[1024];//32*32
	fixed a_11_part[1024];//32*32
	fixed w_11_part[9];//3*3
	weight_id = weight_id - 36864;
	conv_back(gradw_11, gradb_11, grada_11, grada_12, L11, L12, grada_11_part, gradw_11_part, grada_12_part, a_11_part, w_11_part, weights + weight_id);
	for (int i = 0; i < 36864; i++) {
		weights[weight_id + i] += -lr * (gradw_11[i]);
	}

	//L0
	fixed grada_0[3072];//3*32*32
	fixed gradw_0[1728];//64*3*3*3
	fixed gradb_0[64] = { 0 };//no bias
	fixed grada_0_part[1024];//32*32
	fixed gradw_0_part[9];//3*3
	//fixed grada_11_part[1024];//32*32
	fixed a_0_part[1024];//32*32
	fixed w_0_part[9];//3*3
	weight_id = weight_id - 1728;
	conv_back(gradw_0, gradb_0, grada_0, grada_11, L0, L11, grada_0_part, gradw_0_part, grada_11_part, a_0_part, w_0_part, weights + weight_id);
	for (int i = 0; i < 1728; i++) {
		weights[weight_id + i] += -lr * (gradw_13[i]+1);
	}

	std::cout << "weight_id: " << weight_id << std::endl;

	return 0.0;
}

