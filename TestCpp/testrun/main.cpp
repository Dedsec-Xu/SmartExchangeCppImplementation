#include <iostream>
#include "time.h"
#include "fpga_module.h"
#include "resnet18_model.h"
#include "train_noSE.h"
#include "unistd.h"
#include <iostream>

//void train_main()
int main()
{
	std::cout << "train the model:" <<std::endl;
	train(1, 0, 0.5, 0, 0.98);

	//cout << "test: "<<endl;
	//float acc=resnet18(1, 3, 11672232, 1);
	//std::cout << "Accuracy: " << 100 * acc << "%" << std::endl;
	system("pause");

}
