#include <iostream>

using namespace std;

#define BF_CE_1 20
#define BF_CE_2 20
#define BF_CE_3 20
#define BF_CE_4 20
#define BF_B_1 20
#define BF_B_2 20
#define BF_B_3 20
#define BF_B_4 20
#define buffersize_x 20
#define buffersize_y 20

#define buffersize_r 20

#define Reshape_Buffer_Size 10000
#define PI 3.141592654
#define reshape_debug 1

#define bmm_debug 0

typedef float fixed1;

int reshape(fixed1 input1[][buffersize_x][buffersize_y], fixed1 output[][buffersize_x][buffersize_y][buffersize_y], int inputdim[], int dim_1, int dim_2, int dim_3, int dim_4);
int bmm(fixed1 input1[][buffersize_x][buffersize_y], fixed1 input2[][buffersize_y][buffersize_r], fixed1 output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[] );
int main()
{
    cout << "Hello world!" << endl;

    fixed1 input1[buffersize_x][buffersize_x][buffersize_y];
    fixed1 input2[buffersize_x][buffersize_y][buffersize_r];
    fixed1 output2[buffersize_x][buffersize_x][buffersize_r];
    int size_C_dim[3] = {8,8,8};
    int input1_dim[3] = {5,5,4};
    int input2_dim[3] = {5,4,6};
    int output2_dim[3] = {1,1,1};
    fixed1 output[buffersize_x][buffersize_x][buffersize_y][buffersize_y];
    int i = 0;
    if(reshape_debug == 1)
    {
        for (int iter_1 = 0; iter_1 < size_C_dim[0]; iter_1++)
        {
            for (int iter_2 = 0; iter_2 < size_C_dim[1]; iter_2++)
            {
                for (int iter_3 = 0; iter_3 < size_C_dim[2]; iter_3++)
                {
                    i++;
                    input1[iter_1][iter_2][iter_3] = (float) i;	//equals to 0
                    cout << input1[iter_1][iter_2][iter_3] << "\t";
                }
                cout << endl;
            }
            cout << endl;
        }
        reshape(input1, output, size_C_dim, 2, -1, 8, 2);
    }
    if(bmm_debug == 1)
    {
        for (int iter_1 = 0; iter_1 < input1_dim[0]; iter_1++)
        {
            for (int iter_2 = 0; iter_2 < input1_dim[1]; iter_2++)
            {
                for (int iter_3 = 0; iter_3 < input1_dim[2]; iter_3++)
                {
                    i++;
                    input1[iter_1][iter_2][iter_3] = (float) i;	//equals to 0
                    cout << "input1" << "[" << iter_1 << "]" << "[" << iter_2 << "]" << "[" << iter_3 << "]" << "=" << input1[iter_1][iter_2][iter_3] << endl;
                }
            }
        }

        for (int iter_1 = 0; iter_1 < input2_dim[0]; iter_1++)
        {
            for (int iter_2 = 0; iter_2 < input2_dim[1]; iter_2++)
            {
                for (int iter_3 = 0; iter_3 < input2_dim[2]; iter_3++)
                {
                    i++;
                    input2[iter_1][iter_2][iter_3] = (float) i;	//equals to 0
                    cout << "input2" << "[" << iter_1 << "]" << "[" << iter_2 << "]" << "[" << iter_3 << "]" << "=" << input2[iter_1][iter_2][iter_3] << endl;
                }
            }
        }
        fixed1 output[buffersize_x][buffersize_x][buffersize_y][buffersize_y];
        for (int iter_1 = 0; iter_1 < 3; iter_1++)
        {
            cout << output2_dim[iter_1] << endl;
        }
        bmm(input1, input2, output2, input1_dim, input2_dim, output2_dim);
        for (int iter_1 = 0; iter_1 < 3; iter_1++)
        {
            cout << output2_dim[iter_1] << endl;
        }
    }
	//
    return 0;
}

int bmm(fixed1 input1[][buffersize_x][buffersize_y], fixed1 input2[][buffersize_y][buffersize_r],
    fixed1 output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[] )
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
                    fixed1 sum = 0.0;
                    for(int iter_4 = 0; iter_4 < input1_dim[2]; iter_4++)
                    {
                        sum += input1[iter_1][iter_2][iter_4]*input2[iter_1][iter_4][iter_3];//C[a,i,j]=sum(r,A[a,i,r]*B[a,r,j])
                        //cout << sum<<endl;
                    }
                    output[iter_1][iter_2][iter_3] = sum;
                    cout <<  output[iter_1][iter_2][iter_3]<<"\t" ;
                }
                cout << endl;
            }
            cout << endl;
            cout << endl;
        }
    }
    else
    {
         cout << "DIM_ERROR" << endl;
    }
}

int reshape(fixed1 input1[][buffersize_x][buffersize_y], fixed1 output[][buffersize_x][buffersize_y][buffersize_y], int inputdim[], int dim_1, int dim_2, int dim_3, int dim_4)
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
    fixed1 Reshape_Buffer[Reshape_Buffer_Size];
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
                    cout <<  output[iter_1][iter_2][iter_3][iter_4]<<"\t" ;
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
                cout << endl;
                if(break_flag == 1)
                {
                    break;
                }
            }
            cout << endl;
            cout << endl;
            if(break_flag == 1)
            {
                break;
            }
        }
        cout << endl;
        cout << endl;
        cout << endl;
        if(break_flag == 1)
        {
            break;
        }
    }
}
