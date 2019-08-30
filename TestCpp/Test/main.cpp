#include <iostream>

using namespace std;

#define BF_CE_1 1024
#define BF_CE_2 1024
#define BF_CE_3 1024
#define BF_CE_4 1024
#define BF_B_1 1024
#define BF_B_2 1024
#define BF_B_3 1024
#define BF_B_4 1024

#define Reshape_Buffer_Size 10000
#define PI 3.141592654
int main()
{
    cout << "Hello world!" << endl;
    return 0;
}

int reshape(fixed input1[][buffersize_x][buffersize_y], fixed output[][buffersize_x][buffersize_y][buffersize_y], int inputdim[], int dim_1, int dim_2, int dim_3, int dim_4)
{
    int input_d1, input_d2, input_d, input_d1X2, in_iter_1, in_iter_2, in_iter_3;
    input_d1 = inputdim[0];
    input_d2 = inputdim[1];
    input_d3 = inputdim[2];
    in_iter_1 = 0;
    in_iter_2 = 0;
    in_iter_3 = 0;
    input_d1X2 = input_d1 * input_d2;
    int Reshape_Size = input_d1 * input_d2 * input_d3;
    int break_flag = 0;
    fixed Reshape_Buffer[Reshape_Buffer_Size];
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

                    current_iter = iter_1 * dim_1 + iter_2 * dim_2 + iter_3 * dim_3 + iter_4 * dim_4;
                    in_iter_3 += 1;
                    if(in_iter_3>input_d3)
                    {
                        in_iter_3 = 0;
                        in_iter_2 += 1;
                        if(in_iter_2>input_d2)
                        {
                            in_iter_2 = 0;
                            in_iter_1 += 1;
                            if(in_iter_1>input_d1)
                            {
                                break_flag = 1;
                                break;
                            }
                        }

                    }
                    output[iter_1][iter_2][iter_3][iter_4] = input1[in_iter_1][in_iter_2][in_iter_3];
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
}
