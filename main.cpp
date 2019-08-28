//
//  main.cpp
//  Cfunction
//
//  Created by SHU WANG on 8/27/19.
//  Copyright © 2019 SHU WANG. All rights reserved.
//
#include <iostream>
using namespace std;

int bmm(int m,int n,int r, int input1[][3], int input2[][4]){
 
    
    int  a[2][3];
    int  b[3][4];
    int  c[m][r];
    a[2][3] = input1[2][3];
    b[3][4] = input2[3][4];
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
        }
        cout << endl << endl;
    }
    
    return c[m][r];
}

//test mode
int main(){
    int m = 2;///the row of martix A/
    int n = 3;///the coloum of martix A/
    int r = 4;///the coloum of martix B/
    int  input1[2][3]= {{1,2,3},{4,5,6}};
    int  input2[3][4]={{1,2,3,4},{5,6,7,8},{1,2,3,4}};
    bmm(m,n,r,input1,input2);
    return 0;
}
