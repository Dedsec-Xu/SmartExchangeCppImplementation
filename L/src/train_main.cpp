#include "models.h"
#include "time.h"
#include "fpga_module.h"
#include "train_models.h"
#include "unistd.h"
using namespace std;

//void train_main()
void main() 
{   
	cout << "train_main" <<endl;
	train(100, 0, 0.5, 0, 0);
	system("pause");

}