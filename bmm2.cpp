
int bmm(int input1[][buffersize_x][buffersize_y], int input2[][buffersize_y][buffersize_r], int output[][buffersize_x][buffersize_r], int dim)
{
    
    
    int a[m][n];
    int b[n][l];
    int c[m][l];
    for(int d=0; d<=dim; d++)// d= dimension
    {
        for (int i= 0;i < m;i++) {
            for (int j = 0;j < r;j++) {
                a[m][n]=input1[d][i][j];
            }
        }
        
        for (int i= 0;i < m;i++) {
            for (int j = 0;j < r;j++) {
                b[m][n]=input2[d][i][j];
            }
        }
        
        ///initial the computed martix/
        for (int i = 0;i < m;i++) {
            for (int j = 0;j < r;j++) {
                c[i][j] = 0;
            }
        }
        ///compute the martix/
        for (int i = 0;i < m;i++) {
            for (int j = 0;j < r;j++) {
                for (int k = 0;k < n;k++) {
                    c[i][j] = c[i][j] + (a[i][k] * b[k][j]);
                }
            }
        }
        /// display the martix/
        // cout << endl << endl << "result："<< endl << endl;
        for (int i= 0;i < m;i++) {
            for (int j = 0;j < r;j++) {
            // cout << c[i][j] << "\t";
            output[d][i][j]=c[i][j];
            }
            // cout << endl << endl;
        }

        
    }
    return 0;
}
    
    
    
    
    
    
    

