#include "fpga_module.h"
#include<math.h>
#include <iostream>


#define BF_WE 1024
#define BF_IN 50
#define BF_OUT 50
#define BF_PART 1024
#define BF_K 7

int conv(bool bn, bool relu, int batch_size, int ch_in, int ch_out, int size_in, int size_out, int kernel_size, int stride, int padding, fixed* conv_in, fixed* weights, fixed* bias, fixed* conv_out){
    int P = padding;
    int id_img, id_ch_in, id_ch_out, id_output_row, id_output_col, id_kernel_row, id_kernel_col;
    int start_col, start_row;
    int input_idx =0;
    int output_idx =0;
    int we_idx =0;
    fixed result;
    fixed input_val;
    fixed we_buffer[BF_WE];
    fixed in_buffer[BF_IN];
    fixed out_buffer[BF_OUT];
    int addr_out[BF_OUT];
    int addr_we[BF_WE];
    const int k_out_1 =size_out * size_out;
    const int k_out_2 = ch_out * k_out_1;
    const int k_out_3 = size_out;
    const int k_we_1 = kernel_size * kernel_size;
    const int k_we_2 = k_we_1 * ch_in;
    const int k_we_3 = k_we_2 * ch_out;
    int ptr_out =0;
    int cnt_out =0;
    int ptr_in =0;
    int ptr_we =0;
    int cnt_we = 0;
    int i;
    id_img = 0;
    for (id_output_row=0;id_output_row<size_out;id_output_row++){
            for (id_output_col=0;id_output_col<size_out;id_output_col++) {
            	we_idx =0;
            	ptr_we = 0;
                for (id_ch_out = 0; id_ch_out < ch_out; id_ch_out++) {
                    //#pragma HLS UNROLL FACTOR=8
                    result = 0;
                    start_row = id_output_row * stride - P;
                    start_col = id_output_col * stride - P;
                    //output_idx =id_img * ch_out * size_out * size_out + id_ch_out * size_out * size_out + id_output_row * size_out + id_output_col;
                    for (id_kernel_row = 0; id_kernel_row < kernel_size; id_kernel_row++) {
                        for (id_kernel_col = 0; id_kernel_col < kernel_size; id_kernel_col++) {
                            if ((start_row + id_kernel_row < 0) || (start_col + id_kernel_col < 0) || (start_row + id_kernel_row >= size_in) || (start_col + id_kernel_col >= size_in)) {
                                    result += 0;
                            }
                            else{
                                for (id_ch_in = 0; id_ch_in < ch_in; id_ch_in++) {
                                    //#pragma HLS UNROLL FACTOR=8
                                    //#pragma HLS loop_tripcount min=c_size max=c_size
                                    //#pragma HLS PIPELINE
                                    input_idx = id_img * ch_in * size_in * size_in + id_ch_in *size_in * size_in + (start_row + id_kernel_row) * size_in + (start_col + id_kernel_col);
                                    input_val = conv_in[input_idx];
                                    //std::cout<<"fuck "<<input_val<<std::endl;
                                    //we_idx = id_ch_out * ch_in * kernel_size * kernel_size + id_ch_in * kernel_size * kernel_size + id_kernel_row * kernel_size + id_kernel_col;
                                    //result += input_val * we_buffer[ptr_we];
                                    result += input_val * weights[we_idx];
                                    we_idx += k_we_1;
                                    if (id_ch_in == ch_in -1) we_idx -= k_we_2;
                                }
                            }
                            we_idx += 1;
                            if (id_kernel_col == kernel_size -1) we_idx -= kernel_size;
                        }
                        we_idx += kernel_size;
                        if (id_kernel_row == kernel_size -1) we_idx -=k_we_1;
                    }
                    result += bias[id_ch_out];
                    addr_out[ptr_out] = output_idx;
                    if (result <0){
						if (relu) {
							out_buffer[ptr_out] = 0;
						}
						else {
							out_buffer[ptr_out] = result;
						}

                    }
                    else{
                    	out_buffer[ptr_out] =result;
                    }
                    ptr_out +=1;
                    cnt_out +=1;
                    if (ptr_out==BF_OUT){
                    	for (i=0;i<BF_OUT;i++){
                    		conv_out[addr_out[i]]=out_buffer[i];
                    	}
                    	ptr_out -= BF_OUT;
                    }
                    if (cnt_out == k_out_2){
                    	for (i=0;i<k_out_2%BF_OUT;i++){
                    		conv_out[addr_out[i]]=out_buffer[i];
                    	}
                    }
                    output_idx += k_out_1;
                    //return 0;
                    we_idx += k_we_2;
                    if (id_ch_out==ch_out-1) {
                    	output_idx -= k_out_2;
                    	we_idx -= k_we_3;
                    }
                }
                output_idx += 1;
                if (id_output_col== size_out-1) output_idx -= k_out_3;
            }
            output_idx += size_out;
        }
	return 0;
}



int fc(bool bn, bool bias_or_not, bool relu, int batch_size, int ch_in, int ch_out, fixed* fc_in, fixed* weights, fixed* bias, fixed* fc_out){
    int i,j;
    fixed result;
    for (i=0;i<ch_out;i++){
        result =0;
        for(j=0;j<ch_in;j++){
            result += fc_in[j]*weights[i*ch_in+j];
        }
        result += bias[i];
        if ((result <0) && (relu==true)){
        	fc_out[i] =0;
        }
        else{
        	fc_out[i] =result;
        }
    }
    return 0;
}

int max_pooling(int batch_size, int ch, int size_in, int size_out, int kernel_size, fixed* pooling_in, fixed* pooling_out){
	//max pooling with stride = 1
    int id_img, id_ch, id_img_row, id_img_col,i,j;
    int output_idx, input_idx;
    fixed max_val;
    fixed input_val;
    id_img=0;
    for (id_ch =0;id_ch<ch;id_ch++){
        for(id_img_row=0;id_img_row < size_out ;id_img_row++){
                for(id_img_col=0; id_img_col < size_out;id_img_col++){
                    output_idx = id_img*ch*size_out*size_out+id_ch*size_out*size_out+id_img_row*size_out+id_img_col;
                    input_idx = id_img*ch*size_in*size_in+id_ch*size_in*size_in+id_img_row*kernel_size*size_in+id_img_col*kernel_size;
                    max_val = pooling_in[input_idx];
                    for (i=0;i<kernel_size;i++){
                        for(j=0;j<kernel_size;j++){
                            input_val = pooling_in[input_idx+i*size_in+j];
                            if (input_val>max_val) max_val=input_val;
                        }
                    }
                    pooling_out[output_idx] = max_val;
                }
            }
    }
    return 0;
}


int get_label(int num_classes, fixed* vec_in){
    int i;
    int label = 0;
    fixed max_val = vec_in[0];
    for (i=0;i<num_classes;i++){
        if (vec_in[i]>max_val){
            max_val = vec_in[i];
            label = i;
        }
    }
    return label;
}

void batch_norm(int batch_size, int channel, int size, fixed* input) {
	int img_id;
	int channel_id;
	for (img_id = 0; img_id < batch_size; img_id++) {
		for (channel_id = 0; channel_id < channel; channel_id++) {
		    float sum = 0;
			for (int row_id = 0; row_id < size; row_id++) {
				for (int col_id = 0; col_id < size; col_id++) {
					int input_idx = img_id*channel*size*size + channel_id*size*size + row_id*size + col_id;
					sum = sum + input[input_idx];
				}
			}
			float mean = sum / (size*size);
			float var_sum = 0;
			for (int row_id = 0; row_id < size; row_id++) {
				for (int col_id = 0; col_id < size; col_id++) {
					int input_idx = img_id*channel*size*size + channel_id*size*size + row_id*size + col_id;
					var_sum = var_sum + (input[input_idx]-mean)*(input[input_idx] - mean);
				}
			}
			float var = var_sum / (size*size);
			float s = pow(var, 0.5);

			for (int row_id = 0; row_id < size; row_id++) {
				for (int col_id = 0; col_id < size; col_id++) {
					int input_idx = img_id*channel*size*size + channel_id*size*size + row_id*size + col_id;
					input[input_idx] = (input[input_idx]-mean)/(s+0.00001);
				}
			}

		}
	}
}


int max_pooling2(int batch_size, int ch, int size_in, int size_out, int kernel_size, fixed* pooling_in, fixed* pooling_out,bool* max_position) {
	//max pooling with stride = 1
	int id_img, id_ch, id_img_row, id_img_col, i, j;
	int output_idx, input_idx;
	fixed max_val;
	fixed input_val;
	id_img = 0;
	for (id_ch = 0; id_ch<ch; id_ch++) {
		for (id_img_row = 0; id_img_row < size_out; id_img_row++) {
			for (id_img_col = 0; id_img_col < size_out; id_img_col++) {
				output_idx = id_img*ch*size_out*size_out + id_ch*size_out*size_out + id_img_row*size_out + id_img_col;
				input_idx = id_img*ch*size_in*size_in + id_ch*size_in*size_in + id_img_row*kernel_size*size_in + id_img_col*kernel_size;
				max_val = pooling_in[input_idx];
				int max_i = 0;
				int max_j = 0;
				for (i = 0; i<kernel_size; i++) {
					for (j = 0; j<kernel_size; j++) {
						input_val = pooling_in[input_idx + i*size_in + j];
						if (input_val > max_val) {
							max_val = input_val;
							max_i = i;
							max_j = j;
						}
					}
				}
				pooling_out[output_idx] = max_val;
				max_position[input_idx + max_i*size_in + max_j]=true;
			}
		}
	}
	return 0;
}


int matrix_mult(int a1, int a2, int b1, int b2, float* A, float* B, float* result) {
	for (int i = 0; i < a1; i++) {
		for (int j = 0; j < b2; j++) {
			int temp = 0;
			for (int a = 0; a < a2; a++) {
				temp += A[i*a2 + a] * B[a*b2 + j];
			}
			result[i*b2 + j] = temp;
		}
	}
	return 0;
}

void matrix_tran(int row, int col, float* ori, float* tran) {
	int temp[2][3];
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			tran[j*row + i] = ori[i*col + j];
		}
	}
}

void conv_back2(float delta_L[],float delta_L_1[],float L_1[], float weights[],float dw[],int ch_L,int ch_L_1,int size_L,int size_L_1,int kernel_size,float alpha) {


	//float l2[18] = { 1,0,1,0,2,0,0,0,1,2,0,0,1,2,2,0,2,0 };

	const int ch_out2 = ch_L;//4
	const int ch_in2 = ch_L_1;//2
	const int kernel_size2 = kernel_size;
	int size_in = size_L_1;//3
	int size_out = size_L;//2
	const int size3 = ch_L*size_L*size_L;
	const int size2 = ch_L_1*size_L_1*size_L_1;


	//float delta2[size2];
	float dx2[BF_IN];
	float weight_matrix2_rot[BF_OUT][BF_IN][BF_K][BF_K] = { 0 };
	int weight_id2 = 0;
	for (int ch_out_id = 0; ch_out_id < ch_out2; ch_out_id++) {
		for (int ch_in_id = 0; ch_in_id < ch_in2; ch_in_id++) {
			for (int row = 0; row < kernel_size2; row++) {
				for (int col = 0; col < kernel_size2; col++) {
					weight_matrix2_rot[ch_out_id][ch_in_id][kernel_size2 - 1 - row][kernel_size2 - 1 - col] = weights[weight_id2];
					weight_id2++;
				}
			}
		}
	}

	float weight_matrix2_rot_vector[BF_OUT][BF_IN][BF_K*BF_K] = { 0 };
	for (int ch_out_id = 0; ch_out_id < ch_out2; ch_out_id++) {
		for (int ch_in_id = 0; ch_in_id < ch_in2; ch_in_id++) {
			for (int row = 0; row < kernel_size2; row++) {
				for (int col = 0; col < kernel_size2; col++) {
					weight_matrix2_rot_vector[ch_out_id][ch_in_id][row*kernel_size2 + col] = weight_matrix2_rot[ch_out_id][ch_in_id][row][col];
				}
			}
		}
	}

	// 1d delta3 to 2d delta3
	float delta3_vector[BF_OUT][BF_PART] = { 0 };
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < size3 / ch_out2; j++) {
			delta3_vector[i][j] = delta_L[i*size3 / ch_out2 + j];
		}
	}

	float conv_result[BF_OUT][BF_IN][BF_PART] = { 0 };
	float bias_temp[1] = { 0 };
	int padding_temp1 = (size_in - 1 + kernel_size2 - size_out) / 2;
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			conv(false, false, 1, 1, 1, size_out, size_in, kernel_size2, 1, padding_temp1, delta3_vector[i], weight_matrix2_rot_vector[i][j], bias_temp, conv_result[i][j]);
		}
	}

	float delta2_vector[BF_IN][BF_PART] = { 0 };

	for (int ch_in_id = 0; ch_in_id < ch_in2; ch_in_id++) {
		for (int ch_out_id = 0; ch_out_id < ch_out2; ch_out_id++) {
			for (int i = 0; i < (size2 / ch_in2); i++) {
				delta2_vector[ch_in_id][i] += conv_result[ch_out_id][ch_in_id][i];

			}
		}
	}
	std::cout << std::endl;
	//std::cout << "delta2_";
	for (int i = 0; i < ch_in2; i++) {
		for (int j = 0; j < (size2 / ch_in2); j++) {
			delta_L_1[i*(size2 / ch_in2) + j] = delta2_vector[i][j];
			//std::cout << delta2_vector[i][j];
		}
	}
	//std::cout << std::endl;
	//for (int i = 0; i < size2; i++) {
	//	if (l2[i] == 0) {
	//		dx2[i] = 0;
	//	}
	//	else {
	//		dx2[i] = 1;
	//	}
	//	delta2[i] = delta2[i] * dx2[i];
	//}
	//std::cout << "input gradient" << std::endl;
	//for (int i = 0; i < 18; i++) {
	//	std::cout << delta_L_1[i] << " ";
	//}
	//std::cout << std::endl;


	//float alpha = 0.5;

	float dw3[BF_OUT][BF_IN][BF_PART] = { 0 };


	int padding_temp2 = kernel_size2 - 1 + size_out - size_in;
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			float b[] = { 0 };
			conv(false, false, 1, 1, 1, size_in, kernel_size2, size_out, 1, padding_temp2, L_1 + j * size_in*size_in, delta_L + i * size_out*size_out, b, dw3[i][j]);
			//cout << "dw: ";
			/*for (int m = 0; m < kernel_size2*kernel_size2; m++) {

				cout << dw3[i][j][m]<< " ";
			}
			cout << endl;*/
		}
	}
	std::cout << std::endl;

	float db3[BF_OUT];
	for (int i = 0; i < ch_out2; i++) {
		db3[i] = 0;
		for (int j = 0; j < size_out*size_out; j++) {
			db3[i] += delta_L[i * size_out*size_out + j];
		}
	}
	std::cout << "weights gradient:" << std::endl;
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			for (int m = 0; m < kernel_size2*kernel_size2; m++) {
				weights[i*ch_in2*kernel_size2*kernel_size2 + j * kernel_size2*kernel_size2 + m] -= alpha * dw3[i][j][m];
				dw[i*ch_in2*kernel_size2*kernel_size2 + j * kernel_size2*kernel_size2 + m]= dw3[i][j][m];

				std::cout << dw3[i][j][m] << " ";
			}

		}
	}
	float bias[BF_OUT];
	for (int i = 0; i < ch_out2; i++) {
		bias[i] = bias[i] - alpha * db3[i];
	}


	//float delta2_[size2] = {0};
	//cout << "delta2_:" <<endl;
	//conv_back2(delta3, delta2_, l2, 2, 4, 2, weights, 18, 16);
	//for (int i = 0; i < 18; i++) {
	//	cout << delta2_[i] << " ";
	//}
	//cout << endl;

	//system("pause");
}

