#include <iostream>
#include "SEConv.h"
#define THRESHOLD 4e-3

using namespace std;

int main()
{
    float L0[65536]; //64*32*32
	float weight[1728]; //64*3*3*3
	float bias_temp64[64] = { 0.0 };
	SEconv(true, true, 1, 3, 64, 32, 32, 3, 1, 2, img, weights + weight_id, bias_temp64, L0, 64, THRESHOLD);

    return 0;
}
