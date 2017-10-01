#include <iostream>
#include <fstream>
#include <ctime>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
using namespace std;


void handleError(cudaError_t error, const char* text) {

	if (error != cudaSuccess) {
		cout << "ERROR " << text << endl;
		exit(1);
	}
}


__global__ void convolve_xdim (unsigned char* input_img, float* img_ptr, unsigned char* conv_img, size_t cols, size_t rows, int conv_idx){

	size_t x = cols * rows;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= x)
		return;

	int offset = 0;
	size_t cidx = 0;
	float load = 0;
	int b = 0;
	
		cidx = idx % cols;						 
		if (cidx == cols)                      
			cidx = 0;


		if(cidx <= conv_idx){                 				 //if the image index is nearer to the left edge of the image
			b = - cidx;
				for( b  ; b <= conv_idx ; b++)
					load += input_img[b] * img_ptr[b];       
			conv_img[idx] = load;                             
			input_img++;                                           
			load = 0;
		}

		else if (cidx >= cols - conv_idx){               
			offset = cols - cidx - 1;						// if close to the right edge of the image
				for( b = - conv_idx  ; b <= offset ; b++)
					load += input_img[b] * img_ptr[b];
			conv_img[idx] = load;                                      
			input_img++;
			load = 0;
		}


		else{                                                                  
			for( b = - conv_idx  ; b <= conv_idx ; b++)
				load += input_img[b] * img_ptr[b];				 //image index far from either edges condition
			conv_img[idx] = load;
			input_img++;
			load = 0;
		}
		input_img -= x;                     
		
	}



__global__ void convolve_ydim (unsigned char* input_img, float* img_ptr, unsigned char* conv_img, size_t cols, size_t rows, int conv_idx){

	int offset = 0;
	size_t y = 0;
	float load = 0;
	int ridx = 0;
	size_t chnl = cols * rows;

	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= chnl)
		return;


		y = index / cols;									 

		if (y <= conv_idx)	{                 				// image index close to the top of the image              
			ridx = - y;
				for( ridx  ; ridx <= conv_idx ; ridx++)
					load += input_img[ridx * cols] * img_ptr[ridx];		
			conv_img[index] = load;								
			input_img++;
			load = 0;
		}
			
		else if( y >= rows - conv_idx ){           				// image index close to the bottom of the image 
			offset = rows - y - 1;
				for( ridx = - conv_idx  ; ridx <= offset ; ridx++)
					load += input_img[ridx * cols] * img_ptr[ridx];
			conv_img[index] = load;
			input_img++;
			load = 0;
		}


		else{                                                                
			for( ridx = - conv_idx  ; ridx <= conv_idx ; ridx++)			//not close to either top or bottom edges
				load += input_img[ridx * cols] * img_ptr[ridx];
			conv_img[index] = load;
			input_img++;
			load = 0;
		}
		input_img -= chnl;                                    

		
	}
	


void RGBChannels (unsigned char* R, unsigned char* G, unsigned char* B, unsigned char* stored_img, size_t pix_cnt){    

	size_t s = 0;
	for(size_t s = 0 ; s < pix_cnt  ; s++){
		s = s * 3;
		stored_img[s]	= R[s];
		stored_img[s+1] = G[s];					//Separating the r, g, b components from the image.
		stored_img[s+2] = B[s];				
	}
}


int main(int argc, char *argv[]){

string inputfile;
	ifstream filename;
	filename.open( argv[1] , std::ios::in | std::ios::binary );   // reading input ppm file

	if(filename.is_open()){

	unsigned int channel = 0;
	inputfile = "";

	filename >> inputfile;

	if(inputfile == "P6")
		channel = 3;

	else{
		cout << "input file format not correct" << endl;
		exit(1);
	}

	string next;
	getline(filename,next);
	getline(filename,next);

	size_t c;
	size_t r;

	filename >> c;   				// #cols
	filename >> r;					//#rows

	getline(filename,next);
	getline(filename,next);

	char data;
	size_t img_size = c * r;

	unsigned char* red = (unsigned char*) malloc ( img_size * sizeof(unsigned char) );
	unsigned char* green = (unsigned char*) malloc ( img_size * sizeof(unsigned char) );
	unsigned char* blue = (unsigned char*) malloc ( img_size * sizeof(unsigned char) );

	for(size_t i = 0 ; i < img_size ; i++){

		filename.get(data);
		red[i] = data;

		filename.get(data);
		green[i] = data;

		filename.get(data);
		blue[i] = data;
	}

/* Mathematical implemenattion of Gaussian window*/

	int sigma = atoi(argv[2]);
	int windw = sigma * 6 + 1;

	float* k = (float*) calloc (windw, sizeof(float));

	int cntr = ceil( windw/2 );
	k += cntr;

	float a = 0;
	float b = 0;
	for(int j = -cntr ; j <= cntr ; j++){

		a = exp( -0.5 * pow( j/(double)sigma, 2 ) );
		b = sigma * sqrt(2*3.14);
		k[j] = a / b;
	}


float* Gaussian;

	cudaMalloc(&Gaussian, windw * sizeof(float));								// allocating memory for gaussian window on GPU
	cudaMemcpy(Gaussian, k, windw * sizeof(float), cudaMemcpyHostToDevice);

	unsigned char* dR;
	unsigned char* dG;
	unsigned char* dB;

	cudaMalloc(&dR, img_size * sizeof(unsigned char));						// allocating memory for red channel & copying it on GPU
	cudaMemcpy(dR, red, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);					
	cudaMalloc(&dG, img_size * sizeof(unsigned char));					// allocating memory for green & copying it channel on GPU
	cudaMemcpy(dG, green, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);	
	cudaMalloc(&dB, img_size * sizeof(unsigned char));					// allocating memory for blue & copying it channel on GPU
	cudaMemcpy(dB, blue, img_size * sizeof(unsigned char), cudaMemcpyHostToDevice);


	unsigned char* d_convR;
	unsigned char* d_convG;
	unsigned char* d_convB;

	cudaMalloc(&d_convR, img_size * sizeof(unsigned char));				// allocating memory for output red channel on GPU
	cudaMalloc(&d_convG, img_size * sizeof(unsigned char));				// allocating memory for output green channel on GPU
	cudaMalloc(&d_convB, img_size * sizeof(unsigned char));				// allocating memory for output blue channel on GPU
	




	double timestart = clock();                     // time starts

	size_t threads = prop.maxThreadsPerBlock;
	size_t blocks = ceil(img_size/threads);

	convolve_xdim<<<blocks,threads>>>(dR, Gaussian, d_convR, c, r, cntr);							//convolving red channel along x and y dimensions	
	convolve_ydim<<<blocks,threads>>>(d_convR, Gaussian, d_convR, c, r, cntr);			

	convolve_xdim<<<blocks,threads>>>(dG, Gaussian, d_convG, c, r, cntr);							//convolving green channel along x and y dimensions
	convolve_ydim<<<blocks,threads>>>(d_convG, Gaussian, d_convG, c, r, cntr);


	convolve_xdim<<<blocks,threads>>>(dB, Gaussian, d_convB, c, r, cntr);					//convolving blue channel along x and y dimensions				
	convolve_ydim<<<blocks,threads>>>(d_convB, Gaussian, d_convB, c, r, cntr);

	unsigned char* h_outputRed = (unsigned char*) malloc( img_size * sizeof(unsigned char));	
	cudaMemcpy(h_outputRed, d_convR, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	unsigned char* h_outputGreen = (unsigned char*) malloc( img_size * sizeof(unsigned char));
	cudaMemcpy(h_outputGreen, d_convG, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);	//Memory allocation on host for output rgb channels	
	unsigned char* h_outputBlue = (unsigned char*) malloc( img_size * sizeof(unsigned char));				
	cudaMemcpy(h_outputBlue, d_convB, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);



	double timestop = clock();							// set the clock to end at this time

	cout << "time taken for GPU execution" << (timestop - timestart) /((double) CLOCKS_PER_SEC)<< " seconds" << endl;		 //Convolution time on GPU in seconds

	unsigned char* h_output = (unsigned char*) malloc (img_size * channel * sizeof(unsigned char));			

	RGBChannels ( h_outputRed, h_outputGreen, h_outputBlue, h_output, img_size);	

	/*stim::image <unsigned char> finalImage (h_output, c, r, channel);										// create an image and save it
	finalImage.save("output.bmp");*/
char* outfile = "output.bmp";
ofstream outfilename;
outfilename.open(outfile, std::ios::out);
if (outfilename.is_open()){
outfilename << c << r << channel;
outfilename << RGBChannels;
}
else { cout<< "error" << endl;}


	cout << endl << "your final image 'output.bmp' is created in your working directory"  << endl;
}

	else {
			cout << "error in opening the file" << endl;
		}

	return 0;
}