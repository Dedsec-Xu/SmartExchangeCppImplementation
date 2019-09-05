#ifndef SE_CONV_MASK_H
#define SE_CONV_MASK_H
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

#define Reshape_Buffer_Size 10000
#define PI 3.141592654

typedef float fixed;

intSEConv(bool bn, bool relu, int batch_size, int ch_in, int ch_out, int size_in, int size_out, int kernel_size, int stride, int padding, float *conv_in, float *weights, float *bias, float *conv_out, int size_splits, float threshold, int size_C_dim[],  int size_B_dim[], float Ce_buffer[], float B_buffer[]);
int reset_parameters(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3], int size_C_dim[], int size_B_dim[]);
int kaiming_uniform_(float input_matrix[][BF_CE_2][BF_CE_3], int dimensions, int size_1, int size_2, int size_3);
int uniform_(float input_matrix[][BF_CE_2][BF_CE_3], float lower_bound, float upper_bound, int size_1, int size_2, int size_3);
int normal_(float input_matrix[][BF_CE_2][BF_CE_3], float mean_in, float std_in, int size_1, int size_2, int size_3);
float gaussrand();
int set_mask();
int get_weight(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3], float weight[][buffersize_x][buffersize_y][buffersize_y]);
int SEforward(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3], float weight[][BF_CE_2][BF_CE_3]);
float SEbackward(float loss, float max_C, float min_C);
float sparsify_and_quantize_C(float qC[][BF_CE_2][BF_CE_3]);
void bmm(float input1[][buffersize_x][buffersize_y], float input2[][buffersize_y][buffersize_r], float output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[] );
int reshape(float input1[][buffersize_x][buffersize_y], float output[][buffersize_x][buffersize_y][buffersize_y], int inputdim[], int dim_1, int dim_2, int dim_3, int dim_4);



#endif