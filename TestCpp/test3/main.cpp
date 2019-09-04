#include <iostream>
#define BF_WE 1024
#define BF_IN 1024
#define BF_OUT 1024
#include <math.h>
using namespace std;
int conv_back(float gradw[], float gradb[10], float grada[],
	float grada_last[], float a[], float a_last[], float grada_part[],
	float gradw_part[], float grada_last_part[], float a_part[], float w_part[], float weights[], int num_weights,
	int ch_out, int num_input, int input_size2, int output_size2);
int conv(bool bn, bool relu, int batch_size, int ch_in, int ch_out, int size_in, int size_out, int kernel_size, int stride, int padding, float *conv_in, float *weights, float *bias, float *conv_out);
int batch_norm(int batch_size, int channel, int size, float *input);

int main()
{
	float grada_43[25] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//512 * 4*4
	float gradw_43[16] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//512 * 512*3 * 3
	float gradb_43[1];	//no bias
	float grada_43_part[25] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//4 * 4
	float gradw_43_part[16] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//3 * 3
	float grada_44_part[25] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//4 * 4
	float a_43_part[25] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//4 * 4
	float w_43_part[16] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};	//3 * 3
	float grada_44[25] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};
	float L43[9] = {0.5532, -0.1883,  0.4587, -1.4525,  1.6249,  1.4031, -0.6604, -0.9478,  0.7610};
	float L44[6] =  {0.0736,  0.3640, 0.9215,  2.1447,-1.2439, 0.3061};
	float weights2[16] = {0, 0, 0.0686, 0, 0, 0.0364, 0, 0, 0, 0.0467, 0, 0, 0, 0, 0, -0.0681};
	cout << "conv back" << endl;
	conv_back(gradw_43, gradb_43, grada_43, grada_44, L43, L44, grada_43_part, gradw_43_part, grada_44_part, a_43_part, w_43_part, weights2, 16, 1, 25, 25, 25);
	return 0;
}

int conv(bool bn, bool relu, int batch_size, int ch_in, int ch_out, int size_in, int size_out,
	int kernel_size, int stride, int padding, float *conv_in, float *weights, float *bias, float *conv_out)
{
	int P = padding;
	int id_img, id_ch_in, id_ch_out, id_output_row, id_output_col, id_kernel_row, id_kernel_col;
	int start_col, start_row;
	int input_idx = 0;
	int output_idx = 0;
	int we_idx = 0;
	float result;
	float input_val;
	float we_buffer[BF_WE];
	float in_buffer[BF_IN];
	float out_buffer[BF_OUT];
	int addr_out[BF_OUT];
	int addr_we[BF_WE];
	const int k_out_1 = size_out * size_out;
	const int k_out_2 = ch_out * k_out_1;
	const int k_out_3 = size_out;
	const int k_we_1 = kernel_size * kernel_size;
	const int k_we_2 = k_we_1 * ch_in;
	const int k_we_3 = k_we_2 * ch_out;
	int ptr_out = 0;
	int cnt_out = 0;
	int ptr_in = 0;
	int ptr_we = 0;
	int cnt_we = 0;
	int i;
	id_img = 0;
	for (id_output_row = 0; id_output_row < size_out; id_output_row++)
	{
		for (id_output_col = 0; id_output_col < size_out; id_output_col++)
		{
			we_idx = 0;
			ptr_we = 0;
			for (id_ch_out = 0; id_ch_out < ch_out; id_ch_out++)
			{
				//#pragma HLS UNROLL FACTOR=8
				result = 0;
				start_row = id_output_row *stride - P;
				start_col = id_output_col *stride - P;
				//output_idx =id_img *ch_out *size_out *size_out + id_ch_out *size_out *size_out + id_output_row *size_out + id_output_col;
				for (id_kernel_row = 0; id_kernel_row < kernel_size; id_kernel_row++)
				{
					for (id_kernel_col = 0; id_kernel_col < kernel_size; id_kernel_col++)
					{
						if ((start_row + id_kernel_row < 0) || (start_col + id_kernel_col < 0) || (start_row + id_kernel_row >= size_in) || (start_col + id_kernel_col >= size_in))
						{
							result += 0;
						}
						else
						{
							for (id_ch_in = 0; id_ch_in < ch_in; id_ch_in++)
							{
								//#pragma HLS UNROLL FACTOR=8
								//#pragma HLS loop_tripcount min=c_size max=c_size
								//#pragma HLS PIPELINE
								input_idx = id_img *ch_in *size_in *size_in + id_ch_in *size_in *size_in + (start_row + id_kernel_row) *size_in + (start_col + id_kernel_col);
								input_val = conv_in[input_idx];
								//std::cout<<"fuck "<<input_val<<std::endl;
								//we_idx = id_ch_out *ch_in *kernel_size *kernel_size + id_ch_in *kernel_size *kernel_size + id_kernel_row *kernel_size + id_kernel_col;
								//result += input_val *we_buffer[ptr_we];
								result += input_val *weights[we_idx];
								we_idx += k_we_1;
								if (id_ch_in == ch_in - 1) we_idx -= k_we_2;
							}
						}

						we_idx += 1;
						if (id_kernel_col == kernel_size - 1) we_idx -= kernel_size;
					}

					we_idx += kernel_size;
					if (id_kernel_row == kernel_size - 1) we_idx -= k_we_1;
				}

				result += bias[id_ch_out];
				addr_out[ptr_out] = output_idx;
				if (result < 0)
				{
					out_buffer[ptr_out] = 0;
				}
				else
				{
					out_buffer[ptr_out] = result;
				}

				ptr_out += 1;
				cnt_out += 1;
				if (ptr_out == BF_OUT)
				{
					for (i = 0; i < BF_OUT; i++)
					{
						conv_out[addr_out[i]] = out_buffer[i];
					}

					ptr_out -= BF_OUT;
				}

				if (cnt_out == k_out_2)
				{
					for (i = 0; i < k_out_2 % BF_OUT; i++)
					{
						conv_out[addr_out[i]] = out_buffer[i];
					}
				}

				output_idx += k_out_1;
				//return 0;
				we_idx += k_we_2;
				if (id_ch_out == ch_out - 1)
				{
					output_idx -= k_out_2;
					we_idx -= k_we_3;
				}
			}

			output_idx += 1;
			if (id_output_col == size_out - 1) output_idx -= k_out_3;
		}

		output_idx += size_out;
	}

	if (bn == 1)
	{
		batch_norm(batch_size, ch_out, size_out, conv_out);
	}

	if (relu == 1)
	{
		for (int i = 0; i < ch_out *size_out * size_out; i++)
		{
			if (conv_out[i] < 0)
			{
				conv_out[i] = 0;
			}
		}
	}

	return 0;
}

int batch_norm(int batch_size, int channel, int size, float *input)
{
	int img_id;
	int channel_id;
	for (img_id = 0; img_id < batch_size; img_id++)
	{
		for (channel_id = 0; channel_id < channel; channel_id++)
		{
			float sum = 0;
			for (int row_id = 0; row_id < size; row_id++)
			{
				for (int col_id = 0; col_id < size; col_id++)
				{
					int input_idx = img_id *channel *size *size + channel_id *size *size + row_id *size + col_id;
					sum = sum + input[input_idx];
				}
			}

			float mean = sum / (size *size);
			float var_sum = 0;
			for (int row_id = 0; row_id < size; row_id++)
			{
				for (int col_id = 0; col_id < size; col_id++)
				{
					int input_idx = img_id *channel *size *size + channel_id *size *size + row_id *size + col_id;
					var_sum = var_sum + (input[input_idx] - mean) *(input[input_idx] - mean);
				}
			}

			float
			var = var_sum / (size *size);
			float s = pow(var, 0.5);

			for (int row_id = 0; row_id < size; row_id++)
			{
				for (int col_id = 0; col_id < size; col_id++)
				{
					int input_idx = img_id *channel *size *size + channel_id *size *size + row_id *size + col_id;
					input[input_idx] = (input[input_idx] - mean) / (s + 0.00001);
				}
			}
		}
	}
}

int conv_back(float gradw[], float gradb[10], float grada[],
	float grada_last[], float a[], float a_last[], float grada_part[],
	float gradw_part[], float grada_last_part[], float a_part[], float w_part[], float weights[], int num_weights,
	int ch_out, int num_input, int input_size2, int output_size2)
{
	// int num_weights = sizeof(gradw) / sizeof(gradw[0]);//2400
	// int ch_out = sizeof(gradb) / sizeof(gradb[0]);//16
	// int num_input = sizeof(grada) / sizeof(grada[0]);//1176
	// int input_size2 = sizeof(grada_part) / sizeof(grada_part[0]);//196
	// int output_size2= sizeof(grada_last_part) / sizeof(grada_last_part[0]);//100
	int input_size = sqrt(input_size2);	//14
	int output_size = sqrt(output_size2);	//10
	int ch_in = num_input / input_size2;	//6
	int num_output = output_size2 * ch_out;	//1600
	int kernel_size = sqrt(num_weights / (ch_out *ch_in));	//5
	//int num_weights;//2400

	cout << input_size2 << endl;

	for (int i = 0; i < ch_out; i++)
	{
		float sum = 0.0;
		for (int j = 0; j < output_size2; j++)
		{
			sum += grada_last[i *output_size2 + j];
		}

		gradb[i] = sum;
	}

	for (int i = 0; i < ch_in; i++)
	{
		for (int k = 0; k < input_size2; k++)
		{
			a_part[k] = a[k + i *input_size2];
		}

		for (int j = 0; j < ch_out; j++)
		{
			for (int p = 0; p < output_size2; p++)
			{
				if (i != 0)
				{;
				}
				else if (1 && (a_last[p + j *input_size2] == 0.0))
				{

					//??????????????????????????????
					grada_last_part[p] = 0.0;
				}
				else
				{
					grada_last_part[p] = grada_last[p + j *input_size2];
				}
			}

			float bias_temp[] = { 0 };
			conv(false, false, 1, 1, 1, input_size, kernel_size, output_size, 1.0, 0, a_part, grada_last_part, bias_temp, gradw_part);
			for (int p = 0; p < kernel_size * kernel_size; p++)
			{
				gradw[j *ch_in *kernel_size *kernel_size + i *kernel_size *kernel_size + p] = grada_part[p];
               // cout << grada[i *input_size2 + p] << endl;
			}


			for (int p = 0; p < kernel_size; p++)
			{
				for (int q = 0; q < kernel_size; q++)
				{
					w_part[p *kernel_size + q] = weights[j *ch_in *kernel_size *kernel_size + i *kernel_size *kernel_size + (kernel_size - 1 - p) *kernel_size + (kernel_size - 1 - q)];
				}
			}

				//cout << gradw[j *ch_in *kernel_size *kernel_size + i *kernel_size *kernel_size + p] << endl;

			conv(false, false, 1, 1, 1, kernel_size, input_size, output_size, 1, 1, w_part, grada_last_part, bias_temp, grada_part);

			for (int p = 0; p < input_size2; p++)
			{
				grada[i *input_size2 + p] += grada_part[p];

			}
		}
	}
	for(int i = 0; i < num_weights; i++)
    {
        cout << gradw[i] << "\t";
	}

}
