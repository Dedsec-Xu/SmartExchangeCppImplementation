#include "models.h"
#include"train_models.h"
#include<random>
void train(int epoch, int resume, float lr, float momentum, float weight_decay){
    int num_weights = 62006;
    int i;
    int start_epoch;
    float best_acc,acc;
    float* weights;
        //weights = (float *) malloc(62006*sizeof(float));
	weights = new float[62006 * sizeof(float)];
    if (resume == 0){
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0,1.0);
        for (i=0;i<62006;i++){
            weights[i]= distribution(generator);
        }
        start_epoch = 0;
        best_acc = 0.0;
    }
    else{
        FILE * f_we = fopen("ckpt_best.bin","rb");
        FILE * f_epoch = fopen("epoch.bin","rb");
        FILE * f_acc = fopen("best_acc.bin","rb");
        fread(weights, sizeof(float), 62006,f_we);
        fread(&start_epoch, sizeof(int), 1, f_epoch);
        fread(&best_acc, sizeof(float), 1, f_acc);
        fclose(f_acc);
        fclose(f_epoch);
        fclose(f_we);
    }
    int print_or_not = 0;
    clock_t t1,t2,t3,t4;
    float t_test, t_train;
    t_test = 0.0;
    FILE * f_we_w;
    FILE * f_epoch_w;
    FILE * f_acc_w;
    for (i=start_epoch;i<start_epoch+epoch;i++){
         std::cout<<"epoch: "<< i <<" training............"<<std::endl;
         t1 = clock();
         train_one_epoch(weights, lr, momentum, weight_decay);
         t2 = clock();
         std::cout<<"epoch: "<< i <<" testing............"<<std::endl;
         t3 = clock();
         acc = test(weights,print_or_not);
         t4 = clock();
         std::cout<< "Test acc is : "<< acc<<std::endl;
         t_test = 1000*(double)(t4 - t3) /(CLOCKS_PER_SEC*20*100);
         t_train = 1000*(double)(t2 - t1) /(CLOCKS_PER_SEC*80*100);
         std::cout<<"The inferecne time for 1 image is "<< t_test<<" ms"<<std::endl;
         std::cout<<"The training time for 1 image is "<< t_train<<" ms"<<std::endl;
         if (acc > best_acc){
              std::cout<<"Saving..............."<<std::endl;
              std::cout<<i<<" "<<acc<<std::endl;
              f_we_w = fopen("ckpt_best.bin","wb");
              f_epoch_w = fopen("epoch.bin","wb");
              f_acc_w = fopen("best_acc.bin","wb");
              fwrite(weights, sizeof(float), 62006,f_we_w);
              fwrite(&i, sizeof(int), 1, f_epoch_w);
              fwrite(&best_acc, sizeof(float), 1, f_acc_w);
              fclose(f_acc_w);
              fclose(f_epoch_w);
              fclose(f_we_w);
         }
    }
    free(weights);
}
int lenet_inference2(fixed* img, fixed* weights){
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

     fixed l2[1176];
	 int l2_idx[4704] = { 0 };
     max_pooling2(1, 6, 28, 14, 2, l1, l2,l2_idx);

     weight_pointer = 456;
     bias_pointer = weight_pointer + 2400;
     fixed l3[1600];
     conv(0, 1, 1, 6, 16, 14, 10, 5, 1, 0, l2, weights + weight_pointer, weights + bias_pointer, l3); 

     fixed l4[400];
	 int l4_idx[1600] = { 0 };
     max_pooling2(1, 16, 10, 5, 2, l3,l4,l4_idx);

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

     int predicted = get_label(10, l7);
     free(img);
     return predicted;
}

float test(fixed * weights,int print_or_not){
     //load binary file for trained weights
     int i,j,k;
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
     for (i=80;i<100;i++){
         for (j=0;j<100;j += 1){
             tmp = std::string(s1)+ std::string(s2)+ std::to_string(i*100 + j)+ std::string(s3);
             img_file = tmp.c_str();
             f_img = fopen(img_file,"rb");
             if (!f_img){
                 std::cout<<"Cannot open the file "<<img_file<<std::endl;
             }
             else{
                 if (print_or_not) std::cout<<"procssing "<<img_file<<"............."<<std::endl;
                 //img = (fixed *) malloc(num_pixels*sizeof(fixed));
				 img = new fixed[num_pixels * sizeof(fixed)];
                 fread(img,sizeof(fixed),num_pixels+1,f_img);
                 label = img[num_pixels];// the label for the image
                 /*
                 for (k=0;k<num_pixels+1;k++){
                     if (k%300==0 or k == num_pixels)std::cout<<img[k]<<std::endl;
                 }
                 */
                 total++;
				 std::cout << "everything is good so far" << std::endl;
                 predicted = lenet_inference2(img,weights);
                 if (print_or_not) std::cout<<label<<" "<<predicted<<std::endl;
                 if (predicted==label){
                     correct++;
                     if (print_or_not) std::cout<<"current accuracy is "<<(float)correct/total<<std::endl;
                 }
                 //free(img);
             }
            fclose(f_img);
         }
     }
     acc = (float) correct/total;
     return acc;

}

float lenet_back(fixed* l0, fixed* weights, float lr, float momentum, float weight_decay){
     int weight_pointer;
     int bias_pointer;
     int i,j,p,k,q;
     // for pytorch, the feature map (4D) is batch_size x channel x row x col
     // for pytorch, the weighs(4D) are ch_out x ch_in x kernel_row x kernel_col
     // for pytorch, the bias(1D) are ch_out
     fixed * we_conv;
     fixed * bias_conv;
     fixed * we_fc;
     fixed * bias_fc;
     fixed sum;
     weight_pointer = 0;
     bias_pointer = weight_pointer + 450;
     fixed l1[4704];
     conv(0, 1, 1, 3, 6, 32, 28, 5, 1, 0, l0, weights + weight_pointer, weights + bias_pointer, l1);

     fixed l2[1176];
     int l2_idx[4704];
     max_pooling2(1, 6, 28, 14, 2, l1, l2,l2_idx);

     weight_pointer = 456;
     bias_pointer = weight_pointer + 2400;
     fixed l3[1600];
     conv(0, 1, 1, 6, 16, 14, 10, 5, 1, 0, l2, weights + weight_pointer, weights + bias_pointer, l3);

     fixed l4[400];
	 int l4_idx[1600] = {0};
     max_pooling2(1, 16, 10, 5, 2, l3, l4,l4_idx);

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

     int predicted = get_label(10, l7);
	 std::cout << "forward is good" << std::endl;

     fixed grada_l7[10];
     cross_entropy(10,l7, grada_l7);
	 //std::cout << "cross entropy is good" << std::endl;

     fixed gradw_l6[840];
     fixed gradb_l6[10];
     fixed grada_l6[84];
     weight_pointer = 61156;
     bias_pointer = weight_pointer + 840;
     for (i=0;i<10;i++){
         gradb_l6[i] = grada_l7[i];
         for (j=0;j<84;j++){
             gradw_l6[i*84+j]= l6[j] * grada_l7[i];
         }
     }
     for (i=0;i<84;i++){
         sum=0.0;
         for (j=0;j<10;j++){
              sum += grada_l7[j] * weights[weight_pointer+j*84+i];
         }
         grada_l6[i]=sum;
     }
     for (i=0;i<10;i++){
         weights[bias_pointer + i] += -lr * gradb_l6[i];
         for (j=0;j<84;j++){
             weights[weight_pointer + i*84+j] += -lr * gradw_l6[i*84+j];
         }
     }
	 //std::cout << "backward l6 is good" << std::endl;
     fixed gradw_l5[10080];
     fixed gradb_l5[84];
     fixed grada_l5[120];
     weight_pointer = 50992;
     bias_pointer = weight_pointer + 10080;
     for (i=0;i<84;i++){
         gradb_l5[i] = grada_l6[i];
         for (j=0;j<120;j++){
             gradw_l5[i*120+j]= l5[j] * grada_l6[i];
         }
     }
     for (i=0;i<120;i++){
         sum=0.0;
         for (j=0;j<84;j++){
              sum += grada_l6[j] * weights[weight_pointer+j*120+i];
         }
         grada_l5[i]=sum;
     }
     for (i=0;i<84;i++){
         weights[bias_pointer + i] += -lr * gradb_l5[i];
         for (j=0;j<120;j++){
             weights[weight_pointer + i*120+j] += -lr * gradw_l5[i*120+j];
         }
     }
	 //std::cout << "backward l5 is good" << std::endl;

     fixed gradw_l4[48000];
     fixed gradb_l4[120];
     fixed grada_l4[400];
     weight_pointer = 2872;
     bias_pointer = weight_pointer + 48000;
     for (i=0;i<120;i++){
         gradb_l4[i] = grada_l5[i];
         for (j=0;j<400;j++){
             gradw_l4[i*400+j]= l4[j] * grada_l5[i];
         }
     }
     for (i=0;i<400;i++){
         sum=0.0;
         for (j=0;j<120;j++){
              sum += grada_l5[j] * weights[weight_pointer+j*400+i];
         }
         grada_l4[i]=sum;
     }
     for (i=0;i<120;i++){
         weights[bias_pointer + i] += -lr * gradb_l4[i];
         for (j=0;j<400;j++){
             weights[weight_pointer + i*400+j] += -lr * gradw_l4[i*400+j];
         }
     }
     fixed grada_l3[1600]={0.0};
     //for (i=0;i<400;i++){
     //    grada_l3[l4_idx[i]]=grada_l4[i];
     //}
	 int count4 = 0;
	 for (i = 0; i < 1600; i++) {
		 if (l4_idx[i] == 1.0) {
			 grada_l3[i] = grada_l4[count4];
			 count4++;
		 }
	 }
     fixed gradw_l2[2400];
     fixed gradb_l2[16]={0.0};
     fixed grada_l2[1176]={0.0};
     fixed a_l2_part[196];
     fixed grada_l3_part[100];
     fixed grada_l2_part[196];
     fixed gradw_l2_part[25];
     fixed w_l2_part[25];
     weight_pointer = 456;
     bias_pointer = weight_pointer + 2400;
     for (i=0;i<16;i++){
         sum = 0.0;
         for (j=0;j<100;j++){
             sum += grada_l3[j];
         }
         gradb_l2[i]=sum;
     }
     for (i=0;i<6;i++){
         for (k=0;k<196;k++){
             a_l2_part[k] = l2[k+i*196];
         }
         for (j=0;j<16;j++){
              for (p=0;p<100;p++){
                  if (i!=0){;}
                  else if (1 && (l3[p+j*196]==0.0)){
                       grada_l3_part[p] = 0.0;
                  }
                  else{
                       grada_l3_part[p] = grada_l3[p+j*196];
                  }
              }
			  fixed bias_temp[] = { 0 };
              conv(false, false, 1 , 1, 1, 14, 5, 10, 1.0 , 0, a_l2_part, grada_l3_part, bias_temp, gradw_l2_part);
			  //std::cout << "backward conv1 is good" << std::endl;
              for (p=0;p<25;p++){
                  gradw_l2[j*150+i*25+p]=gradw_l2_part[p];
              }
              for (p=0;p<5;p++){
                  for (q=0;q<5;q++){
                      w_l2_part[p*5+q]= weights[weight_pointer + j*150 +i*25 +(5-1-p)*5+(5-1-q)];
                  }
              }
              conv(false, false, 1, 1, 1, 5, 14, 10, 1, 1, w_l2_part, grada_l3_part, bias_temp, grada_l2_part);
			  //std::cout << "backward conv2 is good" << std::endl;
              for (p=0;p<196;p++){
                  grada_l2[i*196+p] += grada_l2_part[p];
              }
         }
     }
     for (i=0;i<16;i++){
         weights[bias_pointer + i] += -lr * gradb_l2[i];
     }
     for (i=0;i<2400;i++){
         weights[weight_pointer + i] += -lr * gradw_l2[i];
     }
     fixed grada_l1[4704]={0.0};
     //for (i=0;i<1176;i++){
     //    grada_l1[l2_idx[i]]=grada_l2[i];
     //}
	 int count2 = 0;
	 for (i = 0; i < 4704; i++) {
		 if (l2_idx[i] == 1) {
			 grada_l1[i] = grada_l2[count2];
			 count2++;
		 }
	 }

     fixed gradw_l0[450];
     fixed gradb_l0[6]={0.0};
     fixed grada_l0[3072]={0.0};
     fixed a_l0_part[1024];
     fixed grada_l1_part[784];
     fixed grada_l0_part[1024];
     fixed gradw_l0_part[25];
     fixed w_l0_part[25];
     weight_pointer = 0;
     bias_pointer = weight_pointer + 450;
     for (i=0;i<6;i++){
         sum = 0.0;
         for (j=0;j<784;j++){
             sum += grada_l1[j];
         }
         gradb_l0[i]=sum;
     }
     for (i=0;i<3;i++){
         for (k=0;k<1024;k++){
             a_l0_part[k] = l0[k+i*1024];
         }
         for (j=0;j<6;j++){
              for (p=0;p<784;p++){
                  if (i!=0){;}
                  else if (1 && (l1[p+j*1024]==0.0)){
                       grada_l1_part[p] = 0.0;
                  }
                  else{
                       grada_l1_part[p] = grada_l1[p+j*1024];
                  }
              }
			  fixed bias_temp2[1] = { 0 };
              conv(false, false, 1.0 , 1, 1, 32, 5, 28, 1.0 , 0, a_l0_part, grada_l1_part, bias_temp2, gradw_l0_part);
              for (p=0;p<25;p++){
                  gradw_l0[j*75+i*25+p]=gradw_l0_part[p];
              }
              for (p=0;p<5;p++){
                  for (q=0;q<5;q++){
                      w_l0_part[p*5+q]= weights[weight_pointer + j*75 +i*25 +(5-1-p)*5+(5-1-q)];
                  }
              }
              conv(false, false, 1, 1, 1, 5, 32, 28, 1, 1, w_l0_part, grada_l1_part, bias_temp2, grada_l0_part);
              for (p=0;p<1024;p++){
                  grada_l0[i*1024+p] += grada_l0_part[p];
              }
         }
     }
     for (i=0;i<6;i++){
         weights[bias_pointer + i] += -lr * gradb_l0[i];
     }
     for (i=0;i<450;i++){
         weights[weight_pointer + i] += -lr * gradw_l0[i];
     }
     //free(l0);
	 delete[] l0;
	 l0 = NULL;
	 std::cout << "backward is good" << std::endl;
     return predicted;
}

float train_one_epoch(fixed* weights,float lr, float momentum, float weight_decay){
     //load binary file for trained weights
     int i,j,k;
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
	 total = 0;
     float train_error;
     for (i=0;i<80;i++){
         for (j=0;j<100;j += 1){
             tmp = std::string(s1)+ std::string(s2)+ std::to_string(i*100 + j)+ std::string(s3);
             img_file = tmp.c_str();
             f_img = fopen(img_file,"rb");
             if (!f_img){
                 std::cout<<"Cannot open the file "<<img_file<<std::endl;
             }
             else{
                 std::cout<<"procssing "<<img_file<<"............."<<std::endl;
                 //img = (fixed *) malloc(num_pixels*sizeof(fixed));
				 img = new fixed[num_pixels * sizeof(fixed)];
                 fread(img,sizeof(fixed),num_pixels+1,f_img);
                 total++;
				 std::cout << "everything is good so far1" << std::endl;
                 lenet_back(img,weights, lr, momentum, weight_decay);
                 //free(img);
				 std::cout << "everything is good so far2" << std::endl;
             }
            fclose(f_img);
         }
     }
     return 0.1;

}

void cross_entropy(int size, fixed* l7, fixed* grada_l7) {

}

