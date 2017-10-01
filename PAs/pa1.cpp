#include <iostream>
#include <fstream>
#include <ctime>
#include<cmath>
#include<fstream>
#include<string>

using namespace std;

void convolve_xdim (unsigned char* input_img, float* img_ptr, unsigned char* conv_img, int conv_idx, size_t cols, size_t rows){
	size_t x = cols * rows;
	float load = 0;
	int offset = 0;
	size_t cidx = 0;

	int b = 0;


	for(size_t p = 0 ; p < x ; p++){

		cidx = p % cols;
		if (cidx == cols)
			cidx = 0;


		if(cidx <= conv_idx){                 //if the image index is nearer to the left edge of the image
			b = - cidx;
				for( b  ; b <= conv_idx ; b++)
					load += input_img[b] * img_ptr[b];
			conv_img[p] = load;
			input_img++;
			load = 0;
		}

		else if (cidx >= cols - conv_idx){               // if close to the right edge of the image
			offset = cols - cidx - 1;
				for( b = - conv_idx  ; b <= offset ; b++)
					load += input_img[b] * img_ptr[b];
			conv_img[p] = load;
			input_img++;
			load = 0;
		}


		else{                                                                  //image index far from either edges condition
			for( b = - conv_idx  ; b <= conv_idx ; b++)
				load += input_img[b] * img_ptr[b];
			conv_img[p] = load;
			input_img++;
			load = 0;
		}

	}
	input_img -= x;
}

void convolve_ydim (unsigned char* input_img, float* img_ptr, unsigned char* conv_img, int conv_idx, size_t cols, size_t rows){
	size_t y = 0;
	int offset = 0;
	float load = 0;
	int ridx = 0;
	size_t chnl = cols * rows;

	for(size_t indx = 0 ; indx < chnl ; indx++){

		y = indx / cols;									 

		if (y <= conv_idx)	{                                // image index close to the top of the image
			ridx = - y;
				for( ridx  ; ridx <= conv_idx ; ridx++)
					load += input_img[ridx * cols] * img_ptr[ridx];		
			conv_img[indx] = load;								
			input_img++;											
			load = 0;
		}

		else if( y >= rows - conv_idx ){             // image index close to the bottom of the image
			offset = rows - y - 1;
				for( ridx = - conv_idx  ; ridx <= offset ; ridx++)
					load += input_img[ridx * cols] * img_ptr[ridx];
			conv_img[indx] = load;
			input_img++;
			load = 0;
		}


		else{                                                                // if not close to the edges
			for( ridx = - conv_idx  ; ridx <= conv_idx ; ridx++)
				load += input_img[ridx * cols] * img_ptr[ridx];
			conv_img[indx] = load;
			input_img++;
			load = 0;
		}

	}
	input_img -= chnl;                                           
}


void RGBChannels (unsigned char* R, unsigned char* G, unsigned char* B, unsigned char* stored_img, size_t pix_cnt){    

	size_t s = 0;
	for(size_t s = 0 ; s < pix_cnt  ; s++){
		s = s * 3;
		stored_img[s]	= R[s];
		stored_img[s+1] = G[s];				//Separating the r, g, b components from the image.
		stored_img[s+2] = B[s];
	}
}


int main(int argc, char** argv){


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


	double timestart = clock();



	unsigned char* outputRed = (unsigned char*) calloc ( img_size, sizeof(unsigned char) );

	convolve_xdim (red, k, outputRed, c, r, cntr);
	convolve_ydim (outputRed, k, outputRed, c, r, cntr);




	unsigned char* outputGreen = (unsigned char*) calloc ( img_size, sizeof(unsigned char) );

	convolve_xdim (green, k, outputGreen, c, r, cntr);
	convolve_ydim (outputGreen, k, outputGreen, c, r, cntr);


	unsigned char* outputBlue = (unsigned char*) calloc ( img_size, sizeof(unsigned char) );

	convolve_xdim (blue, k, outputBlue, c, r, cntr);
	convolve_ydim (outputBlue, k, outputBlue, c, r, cntr);


	double timestop = std::clock();


	cout << endl << "convolution took approximately " << (timestop - timestart) /((double) CLOCKS_PER_SEC)<< " seconds" << endl;		 // calculate the convolution time in seconds



	unsigned char* output = (unsigned char*) malloc (img_size * channel * sizeof(unsigned char));
	RGBChannels ( outputRed, outputGreen, outputBlue, output, img_size);

	}

	else {
			cout << "error in opening the file" << endl;
		}
}
