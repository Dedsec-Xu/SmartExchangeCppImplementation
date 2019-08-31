void bmm(fixed input1[][buffersize_x][buffersize_y], fixed input2[][buffersize_y][buffersize_r], 
    fixed output[][buffersize_x][buffersize_r], int input1_dim[], int input2_dim[], int output_dim[] )
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
                    // cout <<  output[iter_1][iter_2][iter_3]<<"\t" ;
                }
                // cout << endl;
            }
            // cout << endl;
            // cout << endl;
        }
    }
    else
    {
        //  cout << "DIM_ERROR" << endl;
    }
}