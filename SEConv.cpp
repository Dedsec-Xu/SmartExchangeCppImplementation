//
//  se_conv_mask.cpp
//  
//
//  Created by LIANGDONG XU on 8/27/19.
//  Copyright © 2019 LIANGDONG XU. All rights reserved.
//
#include "fpga_module.h"
#include "SEConv.h"
#include <math.h>

//stride=1, padding=0, dilation=1, groups=1, bias=False, size_splits=64,
//threshold=5e-3 

SEConv::SEConv() {}

SEConv::SEConv(bool bn_input, bool relu_input, int batch_size_input, int ch_in_input,
	int ch_out_input, int size_in_input, int size_out_input, int kernel_size_input,
	int stride_input, int padding_input, fixed *conv_in_input, fixed *weights_input, fixed *bias_input,
	fixed *conv_out_input, bool bias_input, int size_splits_input, float threshold_input)
{
	//initialization, because C is different from python. You need to input all parameters in order to use this function,
	bn = bn_input;
	relu = relu_input;
	batch_size = batch_size_input;
	ch_in = ch_in_input;
	ch_out = ch_out_input;
	size_in = size_in_input;
	size_out = size_out_input;
	kernel_size = kernel_size_input;
	stride = stride_input;
	padding = padding_input;
	conv_in = conv_in_input;
	weights = weights_input;
	bias = bias_input;
	conv_out = conv_out_input;
	bias = bias_input;
	size_splits = size_splits_input;
	threshold = threshold_input;

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
	SIZE_C_dim[2] = size_B;	//the three dimension size of CE

	size_B_dim[0] = size_C_dim[0];	// self.C.size()[0],
	size_B_dim[1] = size_B;
	SIZE_B_dim[2] = size_B;
	set_mask();

	reset_parameters();
}

void SEConv::reset_parameters()
{
	int n = ch_in;
	kaiming_uniform_(Ce_buffer, 3, size_C_dim[0], size_C_dim[1], size_C_dim[2]);
	// with torch.no_grad():
	//     self.B.normal_(0, 1.0 / math.sqrt(self.size_B))//normal distribution
	normal_(B_buffer, 0, sqrt(size_B))
	// if self.bias is not None:
	//     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
	//     bound = 1 / math.sqrt(fan_in)
	//     init.uniform_(self.bias, -bound, bound)

}

// int SEConv::_calculate_fan_out(fixed input_matrix[][BF_CE_2][BF_CE_3], int numel, int dimensions, int size_1, int size_2)//size_1 = tensor.size(0)
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

int SEConv::kaiming_uniform_(fixed input_matrix[][BF_CE_2][BF_CE_3], int dimensions, int size_1, int size_2, int size_3)
{
	//mode='fan_out', nonlinearity='relu'
	double bound = sqrt(6.0) / sqrt(size_2 *size_3);	//replaced SEConv::_calculate_fan_out()
	uniform_(tensor, -bound, bound, size_1, size_2, size_3);
}

int SEConv::uniform_(fixed input_matrix[][BF_CE_2][BF_CE_3], float lower_bound, float upper_bound, int size_1, int size_2, int size_3)
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

int SEConv::normal_(fixed input_matrix[][BF_CE_2][BF_CE_3], float mean_in, float std_in, int size_1, int size_2, int size_3)
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

fixed SEConv::gaussrand()	//Box-Muller
{
	static fixed U, V;
	static int phase = 0;
	fixed Z;

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
	retrn Z;
}

void SEConv::set_mask()
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

void SEConv::get_weight(fixed Ce_buffer[][BF_CE_2][BF_CE_3], fixed B_buffer[][BF_CE_2][BF_CE_3], fixed weight[][BF_CE_2][BF_CE_3])
{

	sparsify_and_quantize_C(qC);

		// this is obviously wrong
		for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
	{
	    for(int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
	    {
	        for(int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
	        {
	            qC[iter_1][iter_2][iter_3] = qC[iter_1][iter_2][iter_3] *mask_data_buffer[iter_1][iter_2][iter_3];
	        }
	    }
	}
	// requires *bmm sparsify_and_nearestpow2 reshape
}

void SEConv::forward(fixed Ce_buffer[][BF_CE_2][BF_CE_3], fixed B_buffer[][BF_CE_2][BF_CE_3], fixed weight[][BF_CE_2][BF_CE_3])
{
	get_weight(Ce_buffer, B_buffer, weight_buffer);
	conv(bn, relu, batch_size, ch_in, ch_out, size_in, size_out, kernel_size, stride, padding, conv_in, weights, bias, conv_out);

	return weight;
}

fixed SEConv::backward(fixed loss, fixed max_C, fixed min_C)
{
	parsify_and_quantize_C(qC);
	
	for (int iter_1 = 0; iter_1 < size_1; iter_1++)
	{
		for (int iter_2 = 0; iter_2 < size_2; iter_2++)
		{
			for (int iter_3 = 0; iter_3 < size_3; iter_3++)
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

					dC_pow[iter_1][iter_2][iter_3] = ((dC_counter[iter_1][iter_2][iter_3] > 0) ? 1 : -1) * ((qC[iter_1][iter_2][iter_3] > 0) ? 1 : -1);
					dC_mul[iter_1][iter_2][iter_3] = pow(2, dC_pow[iter_1][iter_2][iter_3]);
					if (qC[iter_1][iter_2][iter_3] == 0)
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
fixed sparsify_and_quantize_C(fixed qC[][BF_CE_2][BF_CE_3])
{
	fixed input_sign, input_abs;
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
					nextpow2 = pow(2.0，ceil(log_temp));
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

int bmm(int input1[][buffersize_x][buffersize_y], int input2[][buffersize_y][buffersize_r], 
	int output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[])
{
	int d = input1_dim[0];
	int m = input1_dim[1];
	int r = input1_dim[2];
	int a[m][n];
	int b[n][l];
	int c[m][l];
	for (int d = 0; d <= dim; d++) // d= dimension
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < r; j++)
			{
				a[m][n] = input1[d][i][j];
			}
		}

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < r; j++)
			{
				b[m][n] = input2[d][i][j];
			}
		}

		///initial the computed martix/
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < r; j++)
			{
				c[i][j] = 0;
			}
		}
		///compute the martix/
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < r; j++)
			{
				for (int k = 0; k < n; k++)
				{
					c[i][j] = c[i][j] + (a[i][k] * b[k][j]);
				}
			}
		}
		/// display the martix/
		// cout << endl << endl << "result："<< endl << endl;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < r; j++)
			{
				// cout << c[i][j] << "\t";
				output[d][i][j] = c[i][j];
			}
			// cout << endl << endl;
		}
	}
	return 0;
}