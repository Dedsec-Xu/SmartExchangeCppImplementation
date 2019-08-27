//
//  se_conv_mask.cpp
//  
//
//  Created by LIANGDONG XU on 8/27/19.
//  Copyright © 2019 LIANGDONG XU. All rights reserved.
//
#include <iostream>
using namespace std;
int main()
{

    return 0;
}
/* This is a 2d version of the problem
float * reshape_3 (float * W, int M, int C)//im guessing W is the pointer of row Wi
{
    float (*Mat_p)[3];// p is the reshaped matrix
    int Mat_Col = C / 3;// the row count of mat
    int Mat_Zero = C % 3;// the padded zero count
    int iter;
    //requires memallocate here
    for(iter = 0; iter < 3 * (Mat_Col + 1); iter++)
    {
        if(iter < C)
        {
            Mat_p[iter/3][iter%3] = W[iter];
        }
        else
        {
            Mat_p[iter/3][iter%3] = 0.0f;//padding
        }
    }
    return Mat_p;
}
*/
 