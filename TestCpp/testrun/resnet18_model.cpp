#include "resnet18_model.h"
#include <iostream>



int resnet18_inference(fixed* img, fixed* weights) {
	int weight_id;
	int bias_id;

	fixed L0[65536]; //64*32*32
	weight_id = 0; //64*3*3*3
	fixed bias_temp64[64] = { 0.0 };
	conv(true, true, 1, 3, 64, 32, 32, 3, 1, 2, img, weights + weight_id, bias_temp64, L0);
	delete[] img;
	img = NULL;

	//layer1**************************************************
    //block1
	fixed L11[65536];//64*32*32
	weight_id = weight_id+1728;//64*64*3*3
	conv(true, true, 1, 64, 64, 32, 32, 3, 1, 2, L0, weights + weight_id, bias_temp64, L11);

	fixed L12[65536];//64*32*32
	weight_id = weight_id + 36864;//64*64*3*3
	conv(true, false, 1, 64, 64, 32, 32, 3, 1, 2, L11, weights + weight_id, bias_temp64, L12);

	fixed L12_[65536];//64*32*32
	matrix_add(65536, L12, L0, L12_); //Residual 

	//block2
	fixed L13[65536];//64*32*32
	weight_id = weight_id + 36864;//64*3*3*3
	conv(true, true, 1, 64, 64, 32, 32, 3, 1, 2, L12_, weights + weight_id, bias_temp64, L13);

	fixed L14[65536];//64*32*32
	weight_id = weight_id + 36864;//64*3*3*3
	conv(true, false, 1, 64, 64, 32, 32, 3, 1, 2, L13, weights + weight_id, bias_temp64, L14);

	fixed L14_[65536];//64*32*32
	matrix_add(65536, L14, L12_, L14_); //Residual 

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
	matrix_add(32768, L22,L14_down, L22_); //Residual 

    //block2
	fixed L23[32768];//128*16*16
	weight_id = weight_id + 147456;//128*128*3*3
	conv(true, true, 1, 128, 128, 16, 16, 3, 1, 2, L22_, weights + weight_id, bias_temp128, L23);

	fixed L24[32768];//128*16*16
	weight_id = weight_id + 147456;//128*128*3*3
	conv(true, false, 1, 128, 128, 16, 16, 3, 1, 2, L23, weights + weight_id, bias_temp128, L24);

	fixed L24_[32768];//128*16*16
	matrix_add(32768, L24, L22_, L24_);//Residual 

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
	matrix_add(16384, L32, L24_down, L32_); //Residual 

	//block2
	fixed L33[16384];//256*8*8
	weight_id = weight_id + 589824;//256*256*3*3
	conv(true, true, 1, 256, 256, 16, 8, 3, 1, 2, L32_, weights + weight_id, bias_temp256, L33);

	fixed L34[16384];//256*8*8
	weight_id = weight_id + 589824;//256*256*3*3
	conv(true, false, 1, 256, 256, 16, 8, 3, 1, 2, L33, weights + weight_id, bias_temp256, L34);

	fixed L34_[16384];//256*8*8
	matrix_add(16384, L34, L32_, L34_); //Residual 

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
	conv(true, false, 1, 256, 512, 8, 4, 1, 2, 0, L34_, weight_down_L3,bias_temp512,L34_down);//down sample
	fixed L42_[8192];//512*4*4
	matrix_add(8192, L42, L34_down, L42_); //Residual 

	//block2
	fixed L43[8192];//512*4*4
	weight_id = weight_id + 2359296;//512*512*3*3
	conv(true, true, 1, 512, 512, 8, 4, 3, 1, 2, L42_, weights + weight_id, bias_temp512, L43);

	fixed L44[8192];//512*4*4
	weight_id = weight_id + 2359296;//512*512*3*3
	conv(true, true, 1, 512, 512, 8, 4, 3, 1, 2, L43, weights + weight_id, bias_temp512, L44);

	fixed L44_[8192];//512*4*4
	matrix_add(8192, L44, L42, L44_); //Residual 

	//Avepooling*****************************************************
	fixed L5[512];//512*1*1;
	ave_pooling4x4(512,4,1,L44_,L5);

	//Fully connected
	fixed L6[1000];
	weight_id = weight_id + 2359296;
	bias_id = weight_id + 512000;
	fixed bias_temp1000[1000] = { 0.0 };
	fc(false, true, false, 1, 512, 1000, L5, weights + weight_id, weights+bias_id, L6);

	//get label *****************************************************
	int label = get_label(1000, L6);
	std::cout << bias_id + 1000 << std::endl;
	return label;
}
float resnet18(int num_batch, int batch_size,int num_weights, int print_or_not,fixed weights[]) {
	const int numWeights = 11500200;
	//fixed weights[numWeights] = { 0.0 };

	//FILE *f_we;
	//f_we = fopen("Resnet_weights.bin", "rb");
	//if (!f_we) {
	//	std::cout << "Cannot open weight file" << std::endl;
	//}
	//else {
	//	fread(weights, sizeof(fixed), num_weights, f_we);
	//}
	//fclose(f_we);


	const char* img_file = NULL;
	char* s1 = "test_cifar10_bin/img";
	char* s2 = "_";
	char* s3 = ".bin";
	std::string tmp;
	FILE *f_img;
	const int num_pixels = 3072;
	fixed * img;
	int correct, total, label,predicted;
	correct = 0;
	total = 0;
	float acc;

	for (int i = 0; i < num_batch; i++) {
		for (int j = 0; j < batch_size; j += 1) {
			tmp = std::string(s1) + std::string(s2) + std::to_string(8000+i*batch_size + j) + std::string(s3);
			img_file = tmp.c_str();
			f_img = fopen(img_file, "rb");
			if (!f_img) {
				std::cout << "Cannot open images file " << img_file << std::endl;
			}
			else {
				if (print_or_not) std::cout << "testing using " << img_file << "............." << std::endl;
				//fixed img[num_pixels] = { 0.0 };
				img = new fixed[num_pixels * sizeof(fixed)];
				fread(img, sizeof(fixed), num_pixels + 1, f_img);
				label = img[num_pixels];
				total++;
				predicted = resnet18_inference(img, weights);
				if (print_or_not) std::cout << "label: " << label << " " << "predicted: " << predicted << std::endl;
				
				if (predicted == label) {
					correct++;
					if (print_or_not) std::cout << "current accuracy is " << (float)correct / total << std::endl;
				}
			}
			fclose(f_img);
		}
	}
	acc = (float)correct / total;
	return acc;
}