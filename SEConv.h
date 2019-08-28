#ifndef SE_CONV_MASK_H
#define SE_CONV_MASK_H
#include "fpga_module.h"
#define BF_CE_1 1024
#define BF_CE_2 1024
#define BF_CE_3 1024
#define BF_CE_4 1024
#define BF_B_1 1024
#define BF_B_2 1024
#define BF_B_3 1024
#define BF_B_4 1024
#define PI 3.141592654

class SEConv
{
	public:
	SEConv();
	SEConv(bool bn_input, bool relu_input, int batch_size_input, int ch_in_input,
		int ch_out_input, int size_in_input, int size_out_input, int kernel_size_input,
		int stride_input, int padding_input, fixed *conv_in_input, fixed *weights_input, fixed *bias_input,
		fixed *conv_out_input, bool bias_input, int size_splits_input, float threshold_input);
	void reset_parameters();
	int kaiming_uniform_(fixed input_matrix[][BF_CE_2][BF_CE_3], int dimensions, int size_1, int size_2, int size_3);
	int uniform_(fixed input_matrix[][BF_CE_2][BF_CE_3], float lower_bound, float upper_bound, int size_1, int size_2, int size_3);
	int normal_(fixed input_matrix[][BF_CE_2][BF_CE_3], float mean_in, float std_in, int size_1, int size_2, int size_3);
	fixed gaussrand();
	void set_mask();
	void get_weight(fixed Ce_buffer[][BF_CE_2][BF_CE_3], fixed B_buffer[][BF_CE_2][BF_CE_3], fixed weight[][BF_CE_2][BF_CE_3]);
	void forward(fixed Ce_buffer[][BF_CE_2][BF_CE_3], fixed B_buffer[][BF_CE_2][BF_CE_3], fixed weight[][BF_CE_2][BF_CE_3]);

	bool bn;
	bool relu;
	int batch_size;
	int ch_in;
	int ch_out;
	int size_in;
	int size_out;
	int kernel_size;
	int stride;
	int padding;
	fixed * conv_in;
	fixed * weights;
	fixed * bias;
	fixed * conv_out;
	bool bias;
	int size_splits;
	float threshold;
	int size_B;
	int size_C;
	int num_splits;
	fixed Ce_buffer[BF_CE_1][BF_CE_2][BF_CE_3];	//buffered Ce and B
	fixed B_buffer[BF_B_1][BF_B_2][BF_B_3];
	fixed mask_data_buffer[BF_CE_1][BF_CE_1][BF_CE_1];
	fixed qC_buffer[BF_CE_1][BF_CE_1][BF_CE_1];
	int size_C_dim[3];
	int size_B_dim[3];
}

#endif