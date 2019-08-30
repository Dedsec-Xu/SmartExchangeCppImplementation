#include "models.h"
#include "time.h"
#include <math.h>
#include "fpga_module.h"
#include <iostream>
#include "train_model.h"

//using namespace std;



void train_epoch(int num_batch, int batch_size) {
	float loss_threshold = 0.5;
	float learning_rate = 0.3;
	float output_interval = 2;

	int num_weights = 62006;
	//fixed * weights;
	//weights = new fixed[num_weights * sizeof(fixed)];
	fixed weights[62006];
	init_weight(weights, num_weights);

	

	//loop for multiple images
	const char* img_file = NULL;
	char* s1 = "test_cifar10_bin/img";
	char* s2 = "_";
	char* s3 = ".bin";
	std::string tmp;
	FILE *f_img;
	int num_pixels = 3072;
	fixed * img;
	int label;
	for (int i = 0; i < num_batch; i++) {
		for (int j = 0; j < batch_size; j++) {
			tmp = std::string(s1) + std::string(s2) + std::to_string(i*batch_size + j) + std::string(s3);
			img_file = tmp.c_str();
			f_img = fopen(img_file, "rb");
			if (!f_img) {
				std::cout << "Cannot open the file2 " << img_file << std::endl;
			}
			else {
				std::cout << "procssing " << img_file << "............." << std::endl;
				img = new fixed[num_pixels*sizeof(fixed)];

				fread(img, sizeof(fixed), num_pixels + 1, f_img);
				label = img[num_pixels];// the label for the image
				fixed real[10];
				for (int i = 0; i < 10; i++) {
					if (i == label) {
						real[i] = 1;
					}
					else {
						real[i] = 0;
					}
				}
				fixed l1[4704];//6*28*28
				bool l1_max[4704];
				fixed l2[1176];//6*14*14			
				fixed l3[1600];//16*10*10
				bool l3_max[1600];
				fixed l4[400];//16*5*5			
				fixed l5[120];
				fixed l6[84];
				fixed l7[10];
				forward(img, weights, l7, l6, l5, l4, l3, l2, l1, l1_max, l3_max);
				backward(learning_rate, weights, num_weights, real, img, l7, l6, l5, l4, l3, l2, l1, l1_max, l3_max);
			}
		}
	}
	/*std::cout << "weights: ";
	for (int i = 0; i < 62006; i++) {
		std::cout << " " << weights[i];
	}
	std::cout << std::endl;*/
	FILE *fp;
	if ((fp = fopen("train_weights.bin", "wb")) == NULL) {
		std::cout << "fail to open target file��" << std::endl;
		exit(0);
	}
	else {
		std::cout << "writing trained weights......" << std::endl;
	}
	if (fwrite(weights, sizeof(fixed), num_weights, fp) != (num_weights)) {
		std::cout << "fail to write into target file��" << std::endl;
	}
	fclose(fp);
}

void init_weight(fixed*weights,int num_weights) {
	//for (int i = 0; i < num_weights; i++) {
	//	weights[i] = (fixed)rand()/(float)(RAND_MAX)*2-1;
	//	//weights[i] = 0;
	//}
	FILE *f_we;
	f_we = fopen("lenet_weights.bin", "rb");
	if (!f_we) {
		std::cout << "Cannot open the weights" << std::endl;
	}
	else {
		fread(weights, sizeof(fixed), num_weights, f_we);
	}
	fclose(f_we);
}

void forward(fixed* img, fixed* weights, fixed* l7, fixed* l6, fixed* l5, fixed* l4, fixed* l3, fixed* l2, fixed* l1, bool* l1_max, bool* l3_max) {
	int weight_pointer;
	int bias_pointer;
	// for pytorch, the feature map (4D) is batch_size x channel x row x col
	// for pytorch, the weighs(4D) are ch_out x ch_in x kernel_row x kernel_col
	// for pytorch, the bias(1D) are ch_out

	weight_pointer = 0;//
	bias_pointer = weight_pointer + 450;//3*6*5*5 weights  +  6 bias?
	//fixed l1[4704];//6*28*28
	conv(0, 1, 1, 3, 6, 32, 28, 5, 1, 0, img, weights + weight_pointer, weights + bias_pointer, l1);
	delete[] img;
	img = NULL;

	//fixed l2[1176];//6*14*14
	//bool l1_max[4704];
	max_pooling2(1, 6, 28, 14, 2, l1, l2,l1_max);

	weight_pointer = 456;//450+6 ? 
	bias_pointer = weight_pointer + 2400;//6*16*25 weights + 16 bias
	//fixed l3[1600];//16*10*10
	conv(0, 1, 1, 6, 16, 14, 10, 5, 1, 0, l2, weights + weight_pointer, weights + bias_pointer, l3);

	//fixed l4[400];//16*5*5
	//bool l3_max[1600];
	max_pooling2(1, 16, 10, 5, 2, l3, l4,l3_max);

	weight_pointer = 2872;//456+2400+16
	bias_pointer = weight_pointer + 48000;//400*120 weights + 120 bias
	//fixed l5[120];
	fc(0, 1, 1, 1, 400, 120, l4, weights + weight_pointer, weights + bias_pointer, l5);

	weight_pointer = 50992;//2872+48000+120
	bias_pointer = weight_pointer + 10080;//120*84 weights + 84 bias
	//fixed l6[84];
	fc(0, 1, 1, 1, 120, 84, l5, weights + weight_pointer, weights + bias_pointer, l6);

	weight_pointer = 61156;//50992+10080+84
	bias_pointer = weight_pointer + 840;//84*10 weights + 10 bias;
	//fixed l7[10];
	fc(0, 1, 0, 1, 84, 10, l6, weights + weight_pointer, weights + bias_pointer, l7);

	delete[] img;
	img = NULL;

}

void backward(fixed alpha,fixed* weights, int num_weights,  fixed* real, fixed* img,fixed* l7, fixed* l6, fixed* l5, fixed* l4, fixed* l3, fixed* l2, fixed* l1, bool* l1_max, bool* l3_max) {

	//l7*************************************************
	const int size7 = 10;
	//weight_pointer = 61156;
	//bias_pointer = weight_pointer + 840;
	fixed delta7[size7];
	fixed dx7[size7];
	fixed sum = 0;
	for (int i = 0; i < size7; i++) {
		sum += exp(l7[i]);
	}
	
	for (int i = 0; i < size7; i++) {
		delta7[i] = exp(l7[i]) / sum - real[i];
	}
	
	//l6*************************************************
	//weight_pointer = 50992;//2872+48000+120
	//bias_pointer = weight_pointer + 10080;//120*84 weights + 84 bias
	const int size6 = 84;
	fixed delta6[size6];
	fixed dx6[size6];
	fixed weight7_t[size6][size7];
	//weight matrix tranpose
	for (int j = 0; j < size7; j++) {
		for (int i = 0; i < size6; i++) {
			weight7_t[i][j] = weights[61156+j*size7+i];
		}
	}
    //w_l+1_t * delta_l+1
	for (int i = 0; i < size6; i++) {
		delta6[i] = 0;
		for (int j = 0; j < size7; j++) {
			delta6[i] = delta6[i] + weight7_t[i][j] * delta7[j];
		}
		if (l6[i]==0) {
			dx6[i] = 0;
		}
		else {
			dx6[i] = 1;
		}
		delta6[i] = delta6[i] * dx6[i];
	}

	//l5*************************************************
	//weight_pointer = 2872;//456+2400+16
	//bias_pointer = weight_pointer + 48000;//400*120 weights + 120 bias
	const int size5 = 120;
	fixed delta5[size5];
	fixed dx5[size5];
	fixed weight6_t[size5][size6];
	//weight matrix tranpose
	for (int j = 0; j < size6; j++) {
		for (int i = 0; i < size5; i++) {
			weight6_t[i][j] = weights[50992 + j * size6 + i];
		}
	}
	//w_l+1_t * delta_l+1
	for (int i = 0; i < size5; i++) {
		delta5[i] = 0;
		for (int j = 0; j < size6; j++) {
			delta5[i] = delta5[i] + weight6_t[i][j] * delta6[j];
		}
		if (l5[i] == 0) {
			dx5[i] = 0;
		}
		else {
			dx5[i] = 1;
		}
		delta5[i] = delta5[i] * dx5[i];
	}

	//l4*************************************************
	const int size4 = 400;
	fixed delta4[size4];
	fixed dx4[size4];
	fixed weight5_t[size4][size5];
	//weight matrix tranpose
	for (int j = 0; j < size5; j++) {
		for (int i = 0; i < size4; i++) {
			weight5_t[i][j] = weights[2872 + j * size5 + i];
		}
	}
	//w_l+1_t * delta_l+1
	for (int i = 0; i < size4; i++) {
		delta4[i] = 0;
		for (int j = 0; j < size5; j++) {
			delta4[i] = delta4[i] + weight5_t[i][j] * delta5[j];
		}
		if (l4[i] == 0) {
			dx4[i] = 0;
		}
		else {
			dx4[i] = 1;
		}
		delta4[i] = delta4[i] * dx4[i];
	}

	//l3****************************************************
	const int size3 = 1600;
	fixed upsample4[size3];
	fixed delta3[size3];
	int c3 = 0;
	for (int i = 0; i < size3; i++) {
		if (l3_max[i] = true) {
			upsample4[i] = delta4[c3];
			c3++;
		}
		else {
			upsample4[i] = 0;
		}
		delta3[i] = upsample4[i];
	}

	//l2**************************************************
	const int size2 = 1176;//14*14*6
	const int ch_out2 = 16;
	const int ch_in2 = 6;
	const int kernel_size2 = 5;
	fixed delta2[size2];
	fixed dx2[size2];
	fixed weight_matrix2[ch_out2][ch_in2][kernel_size2][kernel_size2] = {0};
	fixed weight_matrix2_rot[ch_out2][ch_in2][kernel_size2][kernel_size2] = { 0 };
	int weight_id2 = 456;
	for (int ch_out_id = 0; ch_out_id < ch_out2; ch_out_id++) {
		for (int ch_in_id = 0; ch_in_id < ch_in2; ch_in_id++) {
			for (int row = 0; row < kernel_size2; row++) {
				for (int col = 0; col < kernel_size2; col++) {
					weight_matrix2[ch_out_id][ch_in_id][row][col] = weights[weight_id2];
					weight_matrix2_rot[ch_out_id][ch_in_id][kernel_size2 - 1 - row][kernel_size2 - 1 - col]=weights[weight_id2];
					weight_id2++;
				}
			}
		}	
	}
    
	fixed weight_matrix2_rot_vector[ch_out2][ch_in2][kernel_size2*kernel_size2] = {0};
	for (int ch_out_id = 0; ch_out_id < ch_out2; ch_out_id++) {
		for (int ch_in_id = 0; ch_in_id < ch_in2; ch_in_id++) {
			for (int row = 0; row < kernel_size2; row++) {
				for (int col = 0; col < kernel_size2; col++) {
					weight_matrix2_rot_vector[ch_out_id][ch_in_id][row*kernel_size2 + col] = weight_matrix2_rot[ch_out_id][ch_in_id][row][col];
				}
			}
		}
	}

	fixed delta3_vector[ch_out2][size3 / ch_out2] = {0};
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < size3 / ch_out2; j++) {
			delta3_vector[i][j] = delta3[i*size3 / ch_out2 + j];
		}
	}

	fixed conv_result[ch_out2][ch_in2][size2 / ch_in2] = { 0 };
	fixed bias_temp[1] = { 0 };
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			conv(0, 0, 1, 1, 1, 10, 14, 5, 1, 8, delta3_vector[i], weight_matrix2_rot_vector[i][j], bias_temp, conv_result[i][j]);
		}
	}

	fixed delta2_vector[ch_in2][size2 / ch_in2] = { 0 };

	for (int ch_in_id = 0; ch_in_id < ch_in2; ch_in_id++) {
		for (int ch_out_id = 0; ch_out_id < ch_out2; ch_out_id++) {
			for (int i = 0; i < (size2 / ch_in2); i++) {
				delta2_vector[ch_in_id][i] += conv_result[ch_out_id][ch_in_id][i];
			}
		}
	}

	for (int i = 0; i < ch_in2; i++) {
		for (int j = 0; j < (size2 / ch_in2); j++) {
			delta2[i*(size2 / ch_in2) + j] = delta2_vector[i][j];
		}
	}

	for (int i = 0; i < size2; i++) {
		if (l2[i] == 0) {
			dx2[i] = 0;
		}
		else {
			dx2[i] = 1;
		}
		delta2[i] = delta2[i]*dx2[i];
	}

	//l1****************************************************
	const int size1 = 4704;
	fixed upsample2[size1];
	fixed delta1[size1];
	int c1 = 0;
	for (int i = 0; i < size1; i++) {
		if (l1_max[i] = true) {
			upsample2[i] = delta2[c1];
			c1++;
		}
		else {
			upsample2[i] = 0;
		}
		delta1[i] = upsample2[i];
	}



	//update weights
	int weight_pointer;
	int bias_pointer;

	//l7
	weight_pointer = 61156;//50992+10080+84
	bias_pointer = weight_pointer + 840;//84*10 weights + 10 bias;
	fixed l6_tran[size6];
	matrix_tran(size6, 1, l6, l6_tran);
	fixed dw7[10 * 84];
	matrix_mult(10, 1, 1, 84, delta7, l6_tran, dw7);

	for (int i = 0; i < 840; i++) {
		weights[61156 + i] = weights[61156 + i] - alpha*dw7[i];
	}
	for (int i = 0; i < 10; i++) {
		weights[61156 + 840 + i] = weights[61156 + 840 + i]-alpha*delta7[i];
	}

	//std::cout << "l7: " << std::endl;
	//for (int i = 0; i < 10; i++) {
	//	std::cout << l7[i] << " ";
	//}
	//std::cout << std::endl;

	//std::cout <<"delta7: "<< std::endl;
	//for (int i = 0; i < 10; i++) {
	//	std::cout << delta7[i]<<" ";
	//}
	//std::cout << std::endl;

	//l6
	weight_pointer = 50992;//2872+48000+120
	bias_pointer = weight_pointer + 10080;//120*84 weights + 84 bias
	fixed l5_tran[120];
	matrix_tran(120, 1, l5, l5_tran);
	fixed dw6[84 * 120];
	matrix_mult(84, 1, 1, 120, delta6, l5_tran, dw6);

	for (int i = 0; i < 84*120; i++) {
		weights[50992 + i] = weights[50992 + i] - alpha*dw6[i];
	}
	for (int i = 0; i < 84; i++) {
		weights[50992 + 10080 + i] = weights[50992 + 10080 + i] - alpha*delta6[i];
	}

	//std::cout << "delta6: " << std::endl;
	//for (int i = 0; i < 84; i++) {
	//	std::cout << delta6[i] << " ";
	//}
	//std::cout << std::endl;

	//l5
	weight_pointer = 2872;//456+2400+16
	bias_pointer = weight_pointer + 48000;//400*120 weights + 120 bias
	fixed l4_tran[400];
	matrix_tran(400, 1, l4, l4_tran);
	fixed dw5[120 * 400];
	matrix_mult(120, 1, 1, 400, delta5, l4_tran, dw5);

	for (int i = 0; i < 48000; i++) {
		weights[2872 + i] = weights[2872 + i] - alpha*dw5[i];
	}
	for (int i = 0; i < 120; i++) {
		weights[2872+48000 + i] = weights[2872 + 48000 + i] - alpha*delta5[i];
	}

	//std::cout << "delta5: " << std::endl;
	//for (int i = 0; i < 120; i++) {
	//	std::cout << delta5[i] << " ";
	//}
	//std::cout << std::endl;


	//l4
	//std::cout << "delta4: " << std::endl;
	//for (int i = 0; i < 400; i++) {
	//	std::cout << delta4[i] << " ";
	//}
	//std::cout << std::endl;

	//l3
	//std::cout << "delta3: " << std::endl;
	//for (int i = 0; i < 1600; i++) {
	//	std::cout << delta3[i] << " ";
	//}
	//std::cout << std::endl;
	weight_pointer = 456;//450+6
	bias_pointer = weight_pointer + 2400;//6*16*25 weights + 16 bias
	fixed dw3[2400];
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 6; j++) {
			fixed b[] = { 0 };
			conv(false, false, 1, 1, 1, 14, 5, 10, 1, 9, l2+j*196, delta3+i*100, b, dw3+i*j*25);
		}
	}

	fixed db3[16];
	for (int i = 0; i < 16; i++) {
		db3[i] = 0;
		for (int j = 0; j < 100; j++) {
			db3[i] += delta3[i * 100 + j];
		}
	}

	for (int i = 0; i < 2400; i++) {
		weights[456 + i] = weights[456 + i] - alpha*dw3[i];
	}

	for (int i = 0; i < 16; i++) {
		weights[456 + 2400] = weights[456 + 2400] - alpha*db3[i];
	}

	//l2
	//std::cout << "delta2: " << std::endl;
	//for (int i = 0; i < 1176; i++) {
	//	std::cout << delta2[i] << " ";
	//}
	//std::cout << std::endl;


	//l1
	weight_pointer = 0;//
	bias_pointer = weight_pointer + 450;//3*6*5*5 weights  +  6 bias

	fixed dw1[450];
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 3; j++) {
			fixed b[] = { 0 };
			conv(false, false, 1, 1, 1, 32, 5, 28, 1, 27, img + j * 1024, delta1 + i * 784, b, dw1 + i*j * 25);
		}
	}

	fixed db1[6];
	for (int i = 0; i < 6; i++) {
		db1[i] = 0;
		for (int j = 0; j < 784; j++) {
			db1[i] += delta1[i * 784 + j];
		}
	}

	for (int i = 0; i < 450; i++) {
		weights[i] = weights[i] - alpha*dw1[i];
	}

	for (int i = 0; i < 6; i++) {
		weights[450] = weights[450] - alpha*db1[i];
	}

	//std::cout << "delta1: " << std::endl;
	//for (int i = 0; i < 4704; i++) {
	//	std::cout << delta1[i] << " ";
	//}
	//std::cout << std::endl;
	
}




