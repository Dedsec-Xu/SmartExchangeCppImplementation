#include "fpga_module.h"
#include<math.h>
#include<iostream>
using namespace std;

int main() {

	float weights_[4][2][2][2] = { { {{-1,0}, {0,2}} ,{{-2,0},{0,1}} },
								   { {{-2,0}, {2,0}} ,{{1,0},{1,0}} },
								   { {{0,1}, {0,-1}} ,{{0,2},{0,2}} },
								   { {{1,0}, {0,0}} ,{{2,2},{2,2}} } };
	float weights[32];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 2; j++) {
			for (int m = 0; m < 2; m++) {
				for (int n = 0; n < 2; n++) {
					weights[i * 8 + j * 4 + m * 2 + n] = weights_[i][j][m][n];
				}
			}
		}
	}
	float delta3[16] = { 1,1,0,2,2,0,0,1,1,0,1,0,2,0,0,2 };
	float l2[18] = { 1,0,1,0,2,0,0,0,1,2,0,0,1,2,2,0,2,0 };
	const int size3 = 16;
	const int size2 = 18;
	const int ch_out2 = 4;
	const int ch_in2 = 2;
	const int kernel_size2 = 2;
	int size_in = 3;
	int size_out = 2;


	float delta2[size2];
	float dx2[size2];
	float weight_matrix2_rot[ch_out2][ch_in2][kernel_size2][kernel_size2] = { 0 };
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

	float weight_matrix2_rot_vector[ch_out2][ch_in2][kernel_size2*kernel_size2] = { 0 };
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
	float delta3_vector[ch_out2][size3 / ch_out2] = { 0 };
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < size3 / ch_out2; j++) {
			delta3_vector[i][j] = delta3[i*size3 / ch_out2 + j];
		}
	}

	float conv_result[ch_out2][ch_in2][size2 / ch_in2] = { 0 };
	float bias_temp[1] = { 0 };
	int padding_temp1 = (size_in - 1 + kernel_size2 - size_out)/2;
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			conv(false, false, 1, 1, 1, size_out, size_in, kernel_size2, 1, padding_temp1, delta3_vector[i], weight_matrix2_rot_vector[i][j], bias_temp, conv_result[i][j]);
		}
	}

	float delta2_vector[ch_in2][size2 / ch_in2] = { 0 };

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
	//for (int i = 0; i < size2; i++) {
	//	if (l2[i] == 0) {
	//		dx2[i] = 0;
	//	}
	//	else {
	//		dx2[i] = 1;
	//	}
	//	delta2[i] = delta2[i] * dx2[i];
	//}
	cout << "input gradient" << endl;
	for (int i = 0; i < 18; i++) {
		cout << delta2[i] << " ";
	}
	cout << endl;


	float alpha = 0.5;

	float dw3[ch_out2][ch_in2][kernel_size2*kernel_size2] = { 0 };

	int padding_temp2 = kernel_size2 - 1 + size_out - size_in;
	cout << padding_temp2 << endl;
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			float b[] = { 0 };
		    conv(false, false, 1, 1, 1,size_in, kernel_size2, size_out, 1,padding_temp2 , l2 + j * size_in*size_in, delta3 + i * size_out*size_out, b, dw3[i][j]);
			//cout << "dw: ";
			/*for (int m = 0; m < kernel_size2*kernel_size2; m++) {

				cout << dw3[i][j][m]<< " ";
			}
			cout << endl;*/
		}
	}
	cout<< endl;

	float db3[ch_out2];
	for (int i = 0; i < ch_out2; i++) {
		db3[i] = 0;
		for (int j = 0; j < size_out*size_out; j++) {
			db3[i] += delta3[i * size_out*size_out + j];
		}
	}
	cout << "weights gradient:" << endl;
	for (int i = 0; i < ch_out2; i++) {
		for (int j = 0; j < ch_in2; j++) {
			for (int m = 0; m < kernel_size2*kernel_size2; m++) {

				weights[i*ch_in2*kernel_size2*kernel_size2 + j * kernel_size2*kernel_size2 + m] -= alpha * dw3[i][j][m];
				cout << dw3[i][j][m]<<" ";
			}
		}
	}
	float bias[ch_out2];
	for (int i = 0; i < ch_out2; i++) {
		bias[i] = bias[i] - alpha * db3[i];
	}

	float dw_[32];
	float delta2_[size2] = {0};
	//cout << "delta2_:" <<endl;
	float delta4[16] = { 1,1,0,2,2,0,0,1,1,0,1,0,2,0,0,2 };

	float l3[18] = { 1,0,1,0,2,0,0,0,1,2,0,0,1,2,2,0,2,0 };
	conv_back2(delta4, delta2_, l3, weights,dw_, 4, 2, 2, 3, 2,0.5);
	/*for (int i = 0; i < 18; i++) {
		cout << delta2_[i] << " ";
	}
	cout << endl;*/

	cout << "dw_ :" << endl;
	for (int i = 0; i < 32; i++) {
		cout << dw_[i] << " ";
	}
	cout << endl;

	system("pause");
}
