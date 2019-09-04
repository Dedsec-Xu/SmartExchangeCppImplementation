#include <iostream>

using namespace std;

int main()
{
    int ch_in = 15;
    int ch_out = 10;
    float gradw[150];
	float gradb[10];
    float grada_last[10] = {0.3,0.2,-0.5,0,0,0,0,0,0,0};
	float grada[15];
    float a[15];
    float weights[150];

    for(int i = 0; i < 15; i++)
    {
        a[i]=i;

    }
    for(int i = 0; i < 150; i++)
    {
        weights[i]=i+1;
    }

	for (int i = 0; i < ch_out; i++) {
		gradb[i] = grada_last[i];
		for (int j = 0; j < ch_in; j++) {
			gradw[i * ch_in + j] = a[j] * grada_last[i];
		}
	}
	for (int i = 0; i < ch_in; i++) {
		int sum = 0.0;
		for (int j = 0; j < ch_out; j++) {
			sum += grada_last[j] * weights[j * ch_in + i];
			//cout << "grada_last[" << j << "] = " << grada_last[j] << endl;
		}
		grada[i] = sum;
		cout << "grada[" << i << "] = " << grada[i] << endl;
	}

    return 0;
}
