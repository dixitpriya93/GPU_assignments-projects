// Convolution_xdim.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <cmath>

using namespace std;

void convolve_x_dim(unsigned char* input_img, unsigned char* convolved_img, float* img_ptr, int conv_idx, size_t cols, size_t rows)
{
	size_t ximg_size = cols * rows;
	float load = 0;
	int offset = 0;
	int cidx = 0;
		
	for (int a = 0; a < ximg_size; a++)
	{
		cidx = a % cols;
		if (cidx == cols)
		{
			cidx == 0;
		}
		if (cidx < conv_idx)						//if the image index is nearer to the left edge of the image 
		{
			int b = -cidx;
			while (b <= conv_idx)
			{
				load += input_img[b] * convolved_img[b];
				b++;
			}
			convolved_img[a] = load;
			input_img++;
			load = 0;
		}
		else if (cidx > cols - conv_idx)			//if the image index is nearer to the right edge of the image
		{
			offset = cols - cidx - 1;
			int b = -conv_idx;
			while (b <= offset)
			{
				load += input_img[b] * convolved_img[b];
				b++;
			}
			convolved_img[a] = load;
			input_img++;
			load = 0;
		}
		else
		{
			int b = -conv_idx;
			while (b <= conv_idx)								// image index far from either edges condition
			{
				load += input_img[b] * convolved_img[b];
				b++;
			}
			convolved_img[a] = load;
			input_img++;
			load = 0;
		}
	}
	input_img = input_img - ximg_size;
}

void convolve_ydim(unsigned char* input_img, unsigned char* convolved_img, float* img_ptr, int conv_idx, size_t cols, size_t rows)
{
	size_t yimg_size = cols * rows;
	float load = 0;
	int offset = 0;
	int ridx = 0;

	for (int c = 0; c < yimg_size; c++)
	{
		ridx = c / cols;
		
		if (ridx < conv_idx)						//if the image index is nearer to the left edge of the image 
		{
			int d = -ridx;
			while (d <= conv_idx)
			{
				load += input_img[d*cols] * convolved_img[d];
				d++;
			}
			convolved_img[c] = load;
			input_img++;
			load = 0;
		}
		else if (ridx > rows - conv_idx)			//if the image index is nearer to the right edge of the image
		{
			offset = rows - ridx - 1;
			int d = -conv_idx;
			while (d <= offset)
			{
				load += input_img[d*cols] * convolved_img[d];
				d++;
			}
			convolved_img[c] = load;
			input_img++;
			load = 0;
		}
		else
		{
			int d = -conv_idx;
			while (d <= conv_idx)								// image index far from either edges condition
			{
				load += input_img[d*cols] * convolved_img[d];
				d++;
			}
			convolved_img[c] = load;
			input_img++;
			load = 0;
		}
	}
	input_img = input_img - yimg_size;

}

void RGB_components(unsigned char * R, unsigned char * G, unsigned char *B, unsigned char *stored_img, int pix_cnt)
{
	int e = 0;
	int idx = 0;
	while (idx < pix_cnt)
	{
		e = 3 * idx;
		stored_img[e] = R[idx];
		stored_img[e + 1] = G[idx];
		stored_img[e + 2] = B[idx];
		idx++;
	}
}

void gaussian_filter(float * kernel)
{
	
}

int main()
{
	
	

    return 0;
}

