int conv2d_weight(float input[], int in_size, int out_size,int in_channels, int out_channels, int weight_size[], float grad_output, int stride, int padding, int dilation, int groups=1)
{
    // Args:
    // input: input tensor of shape (minibatch x in_channels x iH x iW)
    // in_size: input[]array size
    // out_size :grad_output[]array size
    // in_channels: input channel
    // out_channels: output channel
    // weight_size : Shape of the weight gradient tensor
    // grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
    // stride (int or tuple, optional): Stride of the convolution. Default: 1
    // padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    // dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    // groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
    int grad_output_repeat_size = (in_channels / groups)* out_size;
    float grad_output_repeat[BUFFERSIZE];
    for(int i = 0; i < (in_channels / groups); i++)
    {
        for(int j = 0; j < out_size ;i++)
        {
            grad_output_repeat[i * out_size + j] = grad_output[j];
        }
        
    }
    conv(input)

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