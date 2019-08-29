#include "models.h"
#include "time.h"

void test_main()
//void main() 
{
    //test_nn_functions();
    int num_batch=1;
    int batch_size = 100;
    int print_or_not = 1;
    clock_t t1,t2;
    t1 = clock();
    float acc = lenet(num_batch,batch_size,print_or_not);
    t2 = clock();
    std::cout<<"For lenet the test accuracy on CIFAR-10 is "<< 100*acc<<"%"<<std::endl;
    std::cout<<"The inferecne time for 1 image is "<<1000*(double)(t2 - t1) /(CLOCKS_PER_SEC*num_batch*batch_size) <<" ms"<<std::endl;

	getchar();
}
