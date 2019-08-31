#include "fpga_module.h"
#include "resnet18_model.h"
//#include "train_noSE.h"
#include <iostream>
#include <cmath>
using namespace std;

void verify_func(){
//void main() {
	float input[3][3][3] = { {{2,5,6},{3,8,9},{4,7,5}},
						     {{7,5,2},{1,9,1},{3,7,5}},
						     {{2,2,4},{3,3,6},{4,4,8}}
	                        };

	float input_[27];
	cout << "input: " << endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				cout << input[i][j][k] << " ";
				input_[i * 9 + j * 3 + k] = input[i][j][k];
			}
			cout << endl;
		}
		cout << endl;
	}

	//for (int i = 0; i < 27; i++) {
	//	cout << input_[i] << " ";
	//}
	//cout << endl;

	float weights[2][3][3][3] = {{{{1,1,1},{1,1,1},{1,1,1}},
								  {{2,2,2},{2,2,2},{2,2,2}},
								  {{3,3,3},{3,3,3},{3,3,3}}},
		                         
								 {{{1,2,3},{1,2,3},{1,2,3}},
								  {{4,5,6},{4,5,6},{4,5,6}},
								  {{7,8,9},{7,8,9},{7,8,9}}}
								 };
	float weights_[54];
	cout << "weights: " << endl;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int m = 0; m < 3; m++) {
					cout << weights[i][j][k][m]<<" ";
					weights_[i * 27 + j * 9 + k * 3 + m] = weights[i][j][k][m];
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
		cout << endl;
	}
	float bias[2] = { 1,2 };
	float output_[18];
	conv(false, false, 1, 3, 2, 3, 3, 3, 1, 2, input_, weights_, bias, output_);
	float output[2][3][3];
	
	cout << "output: " << endl;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				
				output[i][j][k]=output_[i * 9 + j * 3 + k];
				cout << output[i][j][k] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	float f1[6] = { 1,5,4,3,2,6 };
	float w[3];
	float f2[3][6] = { {1,4,7,9,0,2},
					   {3,2,4,4,3,2},
					   {8,7,3,1,2,2} };


	float arr[10] = { 1,2,3,4,5,6,7,8,9,10 };

	cout << "length of array: " << sizeof(arr) / sizeof(arr[0]) << endl;

	float square = 16;
	cout << "square root of 16: "<<sqrt(square)<<endl;


	system("pause");
}