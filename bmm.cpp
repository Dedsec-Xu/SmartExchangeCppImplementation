//
//  main.cpp
//  Cfunction
//
//  Created by SHU WANG on 8/28/19.
//  Copyright © 2019 SHU WANG. All rights reserved.
//
#include <iostream>
#include "stdio.h"
using namespace std;
int i,x,y,r;
int m=0,n=0,l=0;
int output[4][5];//需要把输出矩阵作为全局变量且固定大小
int bmm( int input1[i][x][y], int input2[i][y][r])
{
  
for(int d=0; d<=i; d++)// d= dimension
{
    int a[m][n];
    int b[n][l];
    int c[m][l];
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
    cout << endl << endl << "result："<< endl << endl;
    for (int i= 0;i < m;i++) {
        for (int j = 0;j < r;j++) {
        cout << c[i][j] << "\t";
           output[x][y]=c[i][j];
        }
        cout << endl << endl;
    }
    
    cout << endl << endl << "result："<< endl << endl;
    for (int i= 0;i < m;i++) {
        for (int j = 0;j < r;j++) {
            cout << c[i][j] << "\t";
        }
        cout << endl << endl;
    }
    
    
}
  return 0;
}
    
    
    
    
    
    
    
//test mode
    int main(){
    //int m = 2;///the row of martix A/
    //int n = 3;///the coloum of martix A/
    //int r = 4;///the coloum of martix B/
    int  input1[2][3][0];
    int  input2[3][4][0];
        printf("1");
    //int bmm(int m,int n,int r, int input1[1][3][4], int input2[1][4][4]);
    //bmm(m,n,r,input1[],input2[]);
    return 0;
     }
