#include "time.h"
#include "fpga_module.h"
#include "train_models.h"
#include "resnet18_model.h"
#include "unistd.h"
using namespace std;

//void train_main()
void main() 
{   
	cout << "train_main" <<endl;
	train(1, 0, 0.5, 0, 0.98);

	//cout << "test: "<<endl;
	//float acc=resnet18(1, 3, 11672232, 1);
	//std::cout << "Accuracy: " << 100 * acc << "%" << std::endl;
	system("pause");

}