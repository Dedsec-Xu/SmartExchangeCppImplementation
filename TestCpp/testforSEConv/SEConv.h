#include <math.h>
#include "fpga_module.h"
#define BF_CE_1 100
#define BF_CE_2 100
#define BF_CE_3 100
#define BF_CE_4 100
#define BF_B_1 100
#define BF_B_2 100
#define BF_B_3 100
#define BF_B_4 100
#define buffersize_x 100
#define buffersize_y 100
#define buffersize_r 100

#define Reshape_Buffer_Size 10000
#define PI 3.141592654

typedef float fixed;

int SEConv(bool bn, bool relu, int batch_size, int ch_in,
	int ch_out, int size_in, int size_out, int kernel_size,
	int stride, int padding, float *conv_in, float *weights, float *bias,
	float *conv_out, int size_splits, float threshold, int size_C_dim[],
	int size_B_dim[], float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_B_2][BF_B_3]);

int reset_parameters(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3],
	int size_C_dim[], int size_B_dim[], int ch_in);

int kaiming_uniform_(float input_matrix[][BF_CE_2][BF_CE_3], int dimensions, int size_1,
    int size_2, int size_3);

int uniform_(float input_matrix[][BF_CE_2][BF_CE_3], float lower_bound, float upper_bound, int size_1, int size_2, int size_3);

int normal_(float input_matrix[][BF_CE_2][BF_CE_3], float mean_in, float std_in, int size_1, int size_2, int size_3);

float gaussrand();

int set_mask(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3], int size_C_dim[], int size_B_dim[], float mask_data_buffer[][BF_CE_2][BF_CE_3]);

int get_weight(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3],
	 float weight[][buffersize_x][buffersize_y][buffersize_y], int size_C_dim[], int size_B_dim[],
	 int weight_dim[], float mask_data_buffer[][BF_CE_2][BF_CE_3], int ch_in, int ch_out, int kernel_size,float qC_buffer[][BF_CE_2][BF_CE_3], float threshold);

int SEforward(bool bn, bool relu, int batch_size, int ch_in,
	int ch_out, int size_in, int size_out, int kernel_size,
	int stride, int padding, float *conv_in, float *weights, float *bias,
	float *conv_out, int size_splits, float threshold, int size_C_dim[],
	int size_B_dim[], float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_B_2][BF_B_3],
	float weight_buffer[][buffersize_x][buffersize_y][buffersize_y],int weight_dim[], float mask_data_buffer[][BF_CE_1][BF_CE_1], float qC_buffer[][BF_CE_1][BF_CE_1]);

float SEbackward(float weight_grad[], float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_B_2][BF_B_3],
                 int size_C_dim[], int size_B_dim[], float max_C, float min_C,
                 float Learning_Rate, float threshold);


float sparsify_and_quantize_C(float Ce_buffer[][BF_CE_2][BF_CE_3], float qC[][BF_CE_2][BF_CE_3], int size_C_dim[], float threshold);

void bmm(float input1[][buffersize_x][buffersize_y], float input2[][buffersize_y][buffersize_r],
    float output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[] );

int reshape(float input1[][buffersize_x][buffersize_y], float output[][buffersize_x][buffersize_y][buffersize_y], int inputdim[], int dim_1, int dim_2, int dim_3, int dim_4);

int get_c_d(int weight_decay, float momentum, float dampening, bool nesterov, float Learning_Rate, float d_p[][BF_CE_2][BF_CE_3], int size_C_dim[]);
int reshapeBC(float weight_grad[], int size_C_dim[],float BC[][BF_CE_2][BF_CE_3]);

