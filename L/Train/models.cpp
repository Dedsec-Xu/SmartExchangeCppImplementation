#include "models.h"
#include <iostream>
//using namespace std;
int lenet_inference(fixed* img, fixed* weights){
     int weight_pointer;
     int bias_pointer;
     int i;
     // for pytorch, the feature map (4D) is batch_size x channel x row x col
     // for pytorch, the weighs(4D) are ch_out x ch_in x kernel_row x kernel_col
     // for pytorch, the bias(1D) are ch_out
     fixed * we_conv;
     fixed * bias_conv;
     fixed * we_fc;
     fixed * bias_fc;
     weight_pointer = 0;
     bias_pointer = weight_pointer + 450;
     fixed l1[4704];
     conv(0, 1, 1, 3, 6, 32, 28, 5, 1, 0, img, weights+ weight_pointer, weights + bias_pointer, l1);
     //free(img);
	 //img = NULL;
	 delete[] img;
	 img = NULL;

     fixed l2[1176];
	 int l1_max[4704];
	 max_pooling2(1, 6, 28, 14, 2, l1, l2, l1_max);
     //max_pooling(1, 6, 28, 14, 2, l1, l2);

     weight_pointer = 456;
     bias_pointer = weight_pointer + 2400;
     fixed l3[1600];
     conv(0, 1, 1, 6, 16, 14, 10, 5, 1, 0, l2, weights + weight_pointer, weights + bias_pointer, l3);

     fixed l4[400];
     //max_pooling(1, 16, 10, 5, 2, l3, l4);
	 int l3_max[1600];
	 max_pooling2(1, 16, 10, 5, 2, l3, l4, l3_max);

     weight_pointer = 2872;
     bias_pointer = weight_pointer + 48000;
     fixed l5[120];
     fc(0, 1,1, 1, 400, 120, l4, weights + weight_pointer, weights + bias_pointer, l5);

     weight_pointer = 50992;
     bias_pointer = weight_pointer + 10080;
     fixed l6[84];
     fc(0, 1,1, 1, 120, 84, l5, weights + weight_pointer, weights + bias_pointer, l6);

     weight_pointer = 61156;
     bias_pointer = weight_pointer + 840;
     fixed l7[10];
     fc(0, 1,0, 1, 84, 10, l6, weights + weight_pointer, weights + bias_pointer, l7);

	 //std::cout << "l7: " << std::endl;
	 //for (int i = 0; i < 10; i++) {
		// std::cout << l7[i] << " ";
	 //}
	 //std::cout << std::endl;

     int predicted = get_label(10, l7);
	 //std::cout << "predicted: " << predicted << std::endl;
     return predicted;
}

float lenet(int num_batch, int batch_size, int print_or_not){
     //load binary file for trained weights
     int i,j,k;
     int num_weights = 62006;
     fixed * weights;
     //weights= (fixed *) malloc(62006*sizeof(fixed));
	 weights = new fixed[62006 * sizeof(fixed)];
     FILE *f_we;
     //f_we = fopen("train_weights.bin","rb");
	 f_we = fopen("lenet_weights.bin", "rb");
     if (!f_we){
        std::cout<<"Cannot open the file1"<<std::endl;
     }
     else{
         fread(weights,sizeof(fixed),num_weights,f_we);
         /*
         for (i=0;i<62006+2;i++){
             if (i%10000==0 or i >= 62006)std::cout<<weights[i]<<std::endl;
         */
     }
     fclose(f_we);
     //loop for multiple images
     const char* img_file=NULL;
     char* s1 = "test_cifar10_bin/img";
     char* s2 ="_";
     char* s3 = ".bin";
     std::string tmp;
     FILE *f_img;
     int num_pixels = 3072;
     fixed * img;
     int correct, total, label;
     correct =0;
     total = 0;
     float acc;
     int predicted;
     for (i=0;i<num_batch;i++){
         for (j=0;j<batch_size;j += 1){
             tmp = std::string(s1)+ std::string(s2)+ std::to_string(i*batch_size + j)+ std::string(s3);
             img_file = tmp.c_str();
             f_img = fopen(img_file,"rb");
             if (!f_img){
                 std::cout<<"Cannot open the file2 "<<img_file<<std::endl;
             }
             else{
                 if (print_or_not) std::cout<<"procssing "<<img_file<<"............."<<std::endl;
				 //getchar();
                 //img = (fixed *) malloc(num_pixels*sizeof(fixed));
				 img = new fixed[num_pixels*sizeof(fixed)];

                 fread(img,sizeof(fixed),num_pixels+1,f_img);
                 label = img[num_pixels];// the label for the image
                 /*
                 for (k=0;k<num_pixels+1;k++){
                     if (k%300==0 or k == num_pixels)std::cout<<img[k]<<std::endl;
                 }
                 */
                 total++;
                 predicted = lenet_inference(img,weights);
				 //std::cout <<"predicted:"<< lenet_inference(img, weights) << std::endl;
				 //std::cout << typeid(lenet_inference(img, weights)).name() << std::endl;
				 
				 //predicted = 0;
                 if (print_or_not) std::cout<<"label: "<<label<<" "<<"predicted: "<<predicted<<std::endl;
                 if (predicted==label){
                     correct++;
                     if (print_or_not) std::cout<<"current accuracy is "<<(float)correct/total<<std::endl;
                 }
                 //free(img);
             }
            fclose(f_img);
         }
     }
     delete [] weights;
	 weights = NULL;
     acc = (float) correct/total;
     std::cout<<"The number of weights for lenet (cifar-10) is "<<num_weights<<std::endl;
     return acc;

}

