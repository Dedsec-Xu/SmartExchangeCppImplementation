#include "fpga_module.h"
#include<math.h>


#define BF_WE 1024
#define BF_IN 1024
#define BF_OUT 1024
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
            for (id_output_col = 0;id_output_col<size_out;id_output_col++) {
            	we_idx = 0;
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
                    	out_buffer[ptr_out] =0;
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


int max_pooling2(int batch_size, int ch, int size_in, int size_out, int kernel_size, fixed* pooling_in, fixed* pooling_out,int* max_position) {
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
				max_position[input_idx + max_i*size_in + max_j]=1;
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

float generateGaussianRandom(float mu, float sigma)
{
	const float epsilon = std::numeric_limits<float>::min();
	const float two_pi = 2.0*3.14159265358979323846;

	static float z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
		return z1 * sigma + mu;

	float u1, u2;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}
