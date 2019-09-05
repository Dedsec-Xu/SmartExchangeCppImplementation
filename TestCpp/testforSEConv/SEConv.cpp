//
//  se_conv_mask.cpp
//
//
//  Created by LIANGDONG XU on 8/27/19.
//  Copyright © 2019 LIANGDONG XU. All rights reserved.
//

#include "SEConv.h"
#include <math.h>
#define dC_threshold 0.5
#define switch_bar 7.0
//#define spars_C_threshold 0.005

//stride=1, padding=0, dilation=1, groups=1, bias=False, size_splits=64,
//threshold=5e-3

int SEConv(bool bn, bool relu, int batch_size, int ch_in,
	int ch_out, int size_in, int size_out, int kernel_size,
	int stride, int padding, float *conv_in, float *weights, float *bias,
	float *conv_out, int size_splits, float threshold, int size_C_dim[],
	int size_B_dim[], float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_B_2][BF_B_3])
{
	float mask_data_buffer[BF_CE_1][BF_CE_2][BF_CE_3];
	float weight_buffer[BF_B_1][buffersize_x][buffersize_y][buffersize_y];
	float qC_buffer[BF_CE_1][BF_CE_2][BF_CE_3];
	//initialization, because C is different from python. You need to input all parameters in order to use this function,
	int size_B;
	int size_C;
	int num_splits;
	if (kernel_size > 1)
	{
		size_B = kernel_size;
		if (ch_in < size_splits)
		{
			num_splits = 1;
			size_splits = ch_in;
		}

		if (ch_in >= size_splits)
		{
			num_splits = ch_in / size_splits;
		}
	}
	else
	{
		//VEC_2_SHAPE :
		//4096: (8, 171,),
		//2048: (4, 171,),
		//1024: (2, 171,),
		//512: (1, 171,),
		//256: (1, 86,),
		//128: (1, 43,),
		//64: (1, 22,),

		size_B = 3;
		switch (ch_in)
		{
			case 4096:
				num_splits = 8;
				size_splits = 171;
				break;
			case 2048:
				num_splits = 4;
				size_splits = 171;
				break;
			case 1024:
				num_splits = 2;
				size_splits = 171;
				break;
			case 512:
				num_splits = 1;
				size_splits = 171;
				break;
			case 256:
				num_splits = 1;
				size_splits = 86;
				break;
			case 128:
				num_splits = 1;
				size_splits = 43;
				break;
			case 64:
				num_splits = 1;
				size_splits = 22;
				break;
			default:
				num_splits = 1;
				size_splits = 22;
		}
	}

	//Ce_Buffered
	//self.C = nn.Parameter(torch.Tensor(
	//out_channels *self.num_splits,
	//self.size_splits *self.kernel_size[0], self.size_B)).float()
	//self.B = nn.Parameter(torch.Tensor(
	//self.C.size()[0], self.size_B, self.size_B)).float()
	//self.register_buffer('mask', torch.Tensor(*self.C.size()).float())
	// self.set_mask()

	// self.reset_parameters()
	size_C_dim[0] = ch_out * num_splits;
	size_C_dim[1] = size_splits * kernel_size;
	size_C_dim[2] = size_B;	//the three dimension size of CE

	size_B_dim[0] = size_C_dim[0];	// self.C.size()[0],
	size_B_dim[1] = size_B;
	size_B_dim[2] = size_B;
	set_mask(Ce_buffer,B_buffer,size_C_dim,size_B_dim,mask_data_buffer);

	reset_parameters(Ce_buffer,B_buffer,size_C_dim,size_B_dim, ch_in);
}

int reset_parameters(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3],
	int size_C_dim[], int size_B_dim[], int ch_in)
{
	int n = ch_in;
	kaiming_uniform_(Ce_buffer, 3, size_C_dim[0], size_C_dim[1], size_C_dim[2]);
	// with torch.no_grad():
	//     self.B.normal_(0, 1.0 / math.sqrt(self.size_B))//normal distribution
	normal_(B_buffer, 0, sqrt(size_B_dim[1]),size_B_dim[0],size_B_dim[1],size_B_dim[2]);
	// if self.bias is not None:
	//     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
	//     bound = 1 / math.sqrt(fan_in)
	//     init.uniform_(self.bias, -bound, bound)

}

// int SEConv::_calculate_fan_out(float input_matrix[][BF_CE_2][BF_CE_3], int numel, int dimensions, int size_1, int size_2)//size_1 = tensor.size(0)
// {//we assert that dimension is 3
//     int fan_out;
//    	// if (dimensions == 2)	// Linear
//    	// {
//    	//     int fan_out = size_2;
//    	// }

//    	// else
//    	// {
//    	// int num_output_fmaps = size_2;
//    	// int receptive_field_size = 1
//    	// if (dimensions > 2)
//    	// {
//    	//     receptive_field_size = numel;
//    	//    	//tensor[0][0].numel()
//    	// }

//     fan_out =  size_2 * numel;
//    	// }

//     return fan_out;
// }

int kaiming_uniform_(float input_matrix[][BF_CE_2][BF_CE_3], int dimensions, int size_1, int size_2, int size_3)
{
	//mode='fan_out', nonlinearity='relu'
	double bound = sqrt(6.0) / sqrt(size_2 *size_3);	//replaced SEConv::_calculate_fan_out()
	uniform_(input_matrix, -bound, bound, size_1, size_2, size_3);
}

int uniform_(float input_matrix[][BF_CE_2][BF_CE_3], float lower_bound, float upper_bound, int size_1, int size_2, int size_3)
{
	for (int iter_1 = 0; iter_1 < size_1; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_2; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_3; iter_3++)
			{
				input_matrix[iter_1][iter_2][iter_3] = (((float) rand() / (float)(RAND_MAX)) *(upper_bound - lower_bound)) + lower_bound;
			}
		}
	}
}

int normal_(float input_matrix[][BF_CE_2][BF_CE_3], float mean_in, float std_in, int size_1, int size_2, int size_3)
{
	for (int iter_1 = 0; iter_1 < size_1; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_2; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_3; iter_3++)
			{
				input_matrix[iter_1][iter_2][iter_3] = mean_in + (gaussrand() *std_in);
			}
		}
	}
}

float gaussrand()	//Box-Muller
{
	static float U, V;
	static int phase = 0;
	float Z;

	if (phase == 0)
	{
		U = rand() / (RAND_MAX + 1.0);
		V = rand() / (RAND_MAX + 1.0);
		Z = sqrt(-2.0* log(U)) *sin(2.0 *PI *V);
	}
	else
	{
		Z = sqrt(-2.0* log(U)) *cos(2.0 *PI *V);
	}

	phase = 1 - phase;
	return Z;
}

int set_mask(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3], int size_C_dim[], int size_B_dim[], float mask_data_buffer[][BF_CE_2][BF_CE_3])
{
	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
			{
				if (fabs(Ce_buffer[iter_1][iter_2][iter_3]) < 1e-6)	//equals to 0
				{
					mask_data_buffer[iter_1][iter_2][iter_3] = 0.0;
				}
				else
				{
					mask_data_buffer[iter_1][iter_2][iter_3] = 1.0;
				}
			}
		}
	}
}

// output = input.new_zeros(input.size())
//             input_sign = input.sign()
//             input_abs = input.abs()

//             nnz_idx = input_abs >= threshold
//             input_abs_nnz = input_abs[nnz_idx]

//             nextpow2 = 2 ** input_abs_nnz.log2().ceil()
//             prevpow2 = nextpow2 / 2.0
//             lerr = input_abs_nnz - prevpow2
//             rerr = nextpow2 - input_abs_nnz
//             lbetter = (lerr < rerr).float()
//             # print(prevpow2.size(), nextpow2.size(), lbetter.size())
//             # output_abs_nnz = prevpow2[lbetter] + nextpow2[~lbetter]
//             output_abs_nnz = prevpow2 * lbetter + nextpow2 * (1 - lbetter)

//             output[nnz_idx] = output_abs_nnz * input_sign[nnz_idx]

int get_weight(float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_CE_2][BF_CE_3],
	 float weight[][buffersize_x][buffersize_y][buffersize_y], int size_C_dim[], int size_B_dim[],
	 int weight_dim[], float mask_data_buffer[][BF_CE_2][BF_CE_3], int ch_in, int ch_out, int kernel_size,float qC_buffer[][BF_CE_2][BF_CE_3], float threshold)
{


	sparsify_and_quantize_C(Ce_buffer, qC_buffer, size_C_dim, threshold);

		// this is obviously wrong

	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
	    for(int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
	    {
	        for(int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
	        {
	            qC_buffer[iter_1][iter_2][iter_3] = qC_buffer[iter_1][iter_2][iter_3] *mask_data_buffer[iter_1][iter_2][iter_3];
	        }
	    }
	}
	float BC[buffersize_x][buffersize_x][buffersize_r];
	int BC_dim[3];
	bmm(qC_buffer, B_buffer, BC, size_C_dim, size_B_dim, BC_dim);//BC = torch.bmm(qC, self.B)
	weight_dim[1] = reshape(BC, weight, BC_dim, ch_out, -1, kernel_size, kernel_size);//weight = BC.reshape(self.out_channels, -1, *self.kernel_size)
	//this get dim 2 size
	weight_dim[0] = ch_out;
	weight_dim[2] = kernel_size;
	weight_dim[3] = kernel_size;

	if(kernel_size == 1)
	{
		weight_dim[1] = ch_in;
	}
	// requires *bmm sparsify_and_nearestpow2 reshape
}

int SEforward(bool bn, bool relu, int batch_size, int ch_in,
	int ch_out, int size_in, int size_out, int kernel_size,
	int stride, int padding, float *conv_in, float *weights, float *bias,
	float *conv_out, int size_splits, float threshold, int size_C_dim[],
	int size_B_dim[], float Ce_buffer[][BF_CE_2][BF_CE_3], float B_buffer[][BF_B_2][BF_B_3],
	float weight_buffer[][buffersize_x][buffersize_y][buffersize_y],int weight_dim[], float mask_data_buffer[][BF_CE_1][BF_CE_1], float qC_buffer[][BF_CE_1][BF_CE_1])
{
	get_weight(Ce_buffer, B_buffer, weight_buffer, size_C_dim, size_B_dim, weight_dim,  mask_data_buffer, ch_in, ch_out, kernel_size,qC_buffer,threshold);
	conv(bn, relu, batch_size, ch_in, ch_out, size_in, size_out, kernel_size, stride, padding, conv_in, weights, bias, conv_out);

	return 1;
}

float SEbackward(float weight_grad[], float Ce_buffer[][BF_CE_2][BF_CE_3], float loss, float max_C, float min_C, float qC_buffer[][BF_CE_1][BF_CE_1], int size_C_dim[], float Learning_Rate, float mask_data_buffer[][BF_CE_1][BF_CE_1], float threshold)
{

	float dC_buffer[BF_CE_1][BF_CE_2][BF_CE_3];
	float dC_sign[BF_CE_1][BF_CE_2][BF_CE_3];
	float dC_counter[BF_CE_1][BF_CE_2][BF_CE_3];
	float dC_pow[BF_CE_1][BF_CE_2][BF_CE_3];
	float dC_mul[BF_CE_1][BF_CE_2][BF_CE_3];
	float dC_add;
	sparsify_and_quantize_C(Ce_buffer, qC_buffer, size_C_dim, threshold);
	get_c_d(0,0,0,0,Learning_Rate,dC_buffer,size_C_dim);

	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
			{
				//dC[dC.abs() <= args.dC_threshold] = 0.0
				//dC = optim.get_d(m.C) //?
				if (dC_buffer[iter_1][iter_2][iter_3] <= dC_threshold)
				{
					dC_buffer[iter_1][iter_2][iter_3] = 0;
				}
				dC_sign[iter_1][iter_2][iter_3] = (dC_buffer[iter_1][iter_2][iter_3] > 0) ? 1 : -1;
				dC_counter[iter_1][iter_2][iter_3] += dC_sign[iter_1][iter_2][iter_3];
				//dC_counter.abs() == args.switch_bar
				if (fabs(dC_counter[iter_1][iter_2][iter_3]) == switch_bar)
				{
					//dC_sign = m.dC_counter.sign() * activated.float();

					dC_pow[iter_1][iter_2][iter_3] = ((dC_counter[iter_1][iter_2][iter_3] > 0) ? 1 : -1) * ((qC_buffer[iter_1][iter_2][iter_3] > 0) ? 1 : -1);
					dC_mul[iter_1][iter_2][iter_3] = pow(2, dC_pow[iter_1][iter_2][iter_3]);
					if (qC_buffer[iter_1][iter_2][iter_3] == 0)
					{
						dC_add = mask_data_buffer[iter_1][iter_2][iter_3] * dC_sign[iter_1][iter_2][iter_3] * min_C;
					}
					else
					{
						dC_add = 0.0;
					}
					dC_counter[iter_1][iter_2][iter_3] = 0.0;
				}
				else
				{
					dC_mul[iter_1][iter_2][iter_3] = 1.0;
					dC_add = 0.0;
				}
				Ce_buffer[iter_1][iter_2][iter_3] = Ce_buffer[iter_1][iter_2][iter_3] * dC_mul[iter_1][iter_2][iter_3] + dC_add;
				if (Ce_buffer[iter_1][iter_2][iter_3] > max_C)//clamp
				{
					Ce_buffer[iter_1][iter_2][iter_3] = max_C;
				}
				else if (Ce_buffer[iter_1][iter_2][iter_3] < -max_C)
				{
					Ce_buffer[iter_1][iter_2][iter_3] = -max_C;
				}
			}
		}
	}
	//qC =
	// loss.backward()

	// if args.switch:
	// 	# update Ce matrices using ``Bucket Switching`` scheme
	// 	for name, m in net.named_modules():
	// 		if not hasattr(m, 'mask'):
	// 			continue
	// 		with torch.no_grad():
	// 			qC = m.sparsify_and_quantize_C()
	// 			# grad_C = m.C.grad
	// 			dC = optim.get_d(m.C)
	// 			if dC is None:
	// 				continue
	// 			if args.dC_threshold > 0.0:

	// 				dC[dC.abs() <= args.dC_threshold] = 0.0
	// 			m.C.grad = None
	// 			dC_sign = dC.sign().float()
	// 			# update ``dC_counter``
	// 			m.dC_counter.add_(dC_sign)
	// 			activated = m.dC_counter.abs() == args.switch_bar
	// 			# if activated.any():
	// 			#     print('Ce is updated!!')
	// 			dC_sign = m.dC_counter.sign() * activated.float()
	// 			# Ce non-zero and gradient non-zero
	// 			dC_pow = dC_sign * qC.sign().float()
	// 			dC_mul = 2 ** dC_pow
	// 			# Ce zero (not in the mask) and gradient non-zero
	// 			dC_add = (qC == 0.0).float() * m.mask * dC_sign * args.min_C
	// 			# update C
	// 			new_C = qC.data * dC_mul + dC_add
	// 			if args.max_C is not None:
	// 				new_C.clamp_(-args.max_C, args.max_C)
	// 			m.C.data = new_C
	// 			# reset activated counters to 0
	// 			m.dC_counter[activated] = 0.0
	// 			# m.C.data = sparsify_and_nearestpow2(new_C, args.threshold)

	// optim.step()
}

float sparsify_and_quantize_C(float Ce_buffer[][BF_CE_2][BF_CE_3], float qC[][BF_CE_2][BF_CE_3], int size_C_dim[], float threshold)
{
	float input_sign, input_abs;
	double log_temp, ceil_temp, prevpow2, nextpow2, lerr, rerr;
	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
			{
				if (Ce_buffer[iter_1][iter_2][iter_3] > 0.0)
				{
					input_sign = 1.0;
				}
				else if (Ce_buffer[iter_1][iter_2][iter_3] < 0.0)
				{
					input_sign = 0.0;
				}

				input_abs = fabs(Ce_buffer[iter_1][iter_2][iter_3]);
				if (input_abs >= threshold)
				{
					log_temp = log(input_abs) / log(2);
					nextpow2 = pow(2.0, ceil(log_temp));
					prevpow2 = nextpow2 / 2.0;
					lerr = input_abs - prevpow2;
					rerr = nextpow2 - input_abs;
					if (lerr < rerr)
					{
						qC[iter_1][iter_2][iter_3] = prevpow2;
					}
					else
					{
						qC[iter_1][iter_2][iter_3] = nextpow2;
					}
				}
				else
				{
					qC[iter_1][iter_2][iter_3] = 0.0;
				}
			}
		}
	}
}

// int get_c_d(float weight_buffer[][BF_B_2][BF_B_3], float B_buffer[][BF_B_2][BF_B_3], float Cd_buffer[][BF_B_2][BF_B_3])
// {
// 	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
// 	{
// 		for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
// 		{
// 			for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
// 			{
// 				Cd_buffer[iter_1][iter_2][iter_3] = weight_buffer[iter_1][iter_2][iter_3] / B_buffer[iter_1][iter_2][iter_3];
// 			}
// 		}
// 	}

// }

// int get_b_d(float weight_buffer[][BF_B_2][BF_B_3], float c_buffer[][BF_B_2][BF_B_3], float bd_buffer[][BF_B_2][BF_B_3])
// {
// 	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
// 	{
// 		for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
// 		{
// 			for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
// 			{
// 				bd_buffer[iter_1][iter_2][iter_3] = weight_buffer[iter_1][iter_2][iter_3] / c_buffer[iter_1][iter_2][iter_3];
// 			}
// 		}
// 	}

// }

void bmm(float input1[][buffersize_x][buffersize_y], float input2[][buffersize_y][buffersize_r],
    float output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[] )
{
    if((input1_dim[0] == input2_dim[0])&&(input1_dim[2] == input2_dim[1]))
    {
        output_dim[0] = input1_dim[0];
        output_dim[1] = input1_dim[1];
        output_dim[2] = input2_dim[2];
        int r = input1_dim[2];
        for(int iter_1 = 0; iter_1 < output_dim[0]; iter_1++)
        {
            for(int iter_2 = 0; iter_2 < output_dim[1]; iter_2++)
            {
                for(int iter_3 = 0; iter_3 < output_dim[2]; iter_3++)
                {
                    float sum = 0.0;
                    for(int iter_4 = 0; iter_4 < input1_dim[2]; iter_4++)
                    {
                        sum += input1[iter_1][iter_2][iter_4]*input2[iter_1][iter_4][iter_3];//C[a,i,j]=sum(r,A[a,i,r]*B[a,r,j])
                        //cout << sum<<endl;
                    }
                    output[iter_1][iter_2][iter_3] = sum;
                    // cout <<  output[iter_1][iter_2][iter_3]<<"\t" ;
                }
                // cout << endl;
            }
            // cout << endl;
            // cout << endl;
        }
    }
    else
    {
        //  cout << "DIM_ERROR" << endl;
    }
}

int reshape(float input1[][buffersize_x][buffersize_y], float output[][buffersize_x][buffersize_y][buffersize_y], int inputdim[], int dim_1, int dim_2, int dim_3, int dim_4)
{
    int input_d1, input_d2, input_d3, input_d, input_d1X2, in_iter_1, in_iter_2, in_iter_3;
    input_d1 = inputdim[0];
    input_d2 = inputdim[1];
    input_d3 = inputdim[2];
    in_iter_1 = 0;
    in_iter_2 = 0;
    in_iter_3 = 0;
    input_d1X2 = input_d1 * input_d2;
    int Reshape_Size = input_d1 * input_d2 * input_d3;
    int break_flag = 0;
    float Reshape_Buffer[Reshape_Buffer_Size];
    int current_iter;
    if(dim_2 == -1)
    {
        dim_2 = ((Reshape_Size / dim_1) / dim_3) / dim_4;
    }
    for(int iter_1 = 0; iter_1 < dim_1; iter_1++)
    {
        for(int iter_2 = 0; iter_2 < dim_2; iter_2++)
        {
            for(int iter_3 = 0; iter_3 < dim_3; iter_3++)
            {
                for(int iter_4 = 0; iter_4 < dim_4; iter_4++)
                {
                    output[iter_1][iter_2][iter_3][iter_4] = input1[in_iter_1][in_iter_2][in_iter_3];
                    current_iter = iter_1 * dim_1 + iter_2 * dim_2 + iter_3 * dim_3 + iter_4 * dim_4;
                    // cout << "output" << "[" << iter_1 << "]" << "[" << iter_2 << "]" << "[" << iter_3 << "]" << "[" << iter_4 << "]" << "=" << output[iter_1][iter_2][iter_3][iter_4] << endl;
                    // cout << "input1" << "[" << in_iter_1 << "]" << "[" << in_iter_2 << "]" << "[" << in_iter_3 << "]" << "=" << input1[in_iter_1][in_iter_2][in_iter_3] << endl;
                    in_iter_3 += 1;
                    if(in_iter_3>=input_d3)
                    {
                        in_iter_3 = 0;
                        in_iter_2 += 1;
                        if(in_iter_2>=input_d2)
                        {
                            in_iter_2 = 0;
                            in_iter_1 += 1;
                            if(in_iter_1>=input_d1)
                            {
                                break_flag = 1;
                                break;
                            }
                        }
                    }
                }
                if(break_flag == 1)
                {
                    break;
                }
            }
            if(break_flag == 1)
            {
                break;
            }
        }
        if(break_flag == 1)
        {
            break;
        }
    }
	return dim_2;
}


int reshapeBC(float weight_grad[], int size_C_dim[]，float BC[][BF_CE_2][BF_CE_3])
{
	for(int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
		for(int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
		{
			for(int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
			{
				BC[iter_1][iter_2][iter_3] = weight_grad[iter_1 * size_C_dim[1] * size_C_dim[2] + iter_2 * size_C_dim[2] + iter_3];
			}
		}
	}
}

//int step(int weight_decay, float momentum, float dampening, bool nesterov,int size_C_dim[])
//{
	//(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
//	 d_p = param.grad.data;
// 	if (weight_decay != 0)
// 	{
// 		for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
// 		{
// 			for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
// 			{
// 				for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
// 				{
// 					d_p[iter_1][iter_2][iter_3] = Ce_buffer[iter_1][iter_2][iter_3] + weight_decay;
// 				}
// 			}
// 		}
// 	}

// 	//  if(momentum != 0){
// 	// 	 param_state = self.state[param]
// 	// 	if 'momentum_buffer' not in param_state:
// 	// 		buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
// 	// 	else:
// 	// 		buf = param_state['momentum_buffer']
// 	// 		buf.mul_(momentum).add_(1 - dampening, d_p)
// 	// 	if nesterov:
// 	// 		d_p = d_p.add(momentum, buf)
// 	// 	else:
// 	// 		d_p = buf

// 	//  }

// }

int get_c_d(int weight_decay, float momentum, float dampening, bool nesterov, float Learning_Rate, float d_p[][BF_CE_2][BF_CE_3], int size_C_dim[])
{


	// if weight_decay != 0:
    //                     d_p.add_(weight_decay, param.data)
    //                 if momentum != 0:
    //                     param_state = self.state[param]
    //                     if 'momentum_buffer' not in param_state:
    //                         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
    //                     else:
    //                         buf = param_state['momentum_buffer']
    //                         buf.mul_(momentum).add_(1 - dampening, d_p)
    //                     if nesterov:
    //                         d_p = d_p.add(momentum, buf)
    //                     else:
    //                         d_p = buf

    //                 return -group['lr'] * d_p
	// if (momentum != 0)
	// {
	// 	buf =
	// }
	for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
			{
				d_p[iter_1][iter_2][iter_3] *= -Learning_Rate;
			}
		}
	}



}
