#include <iostream>
#include "error.h"
#include "image.h"

__global__ void kernel(unsigned char* in, unsigned char* out, size_t w, size_t h, int K){
	extern __shared__ unsigned char S[];			//defined a shared memory pointer

	//calculate the index into the output image
	size_t bxi = blockIdx.x * blockDim.x;				//upper left corner of the block
	size_t xi = bxi + threadIdx.x;						//x-index for the current thread
	size_t yi = blockIdx.y * blockDim.y + threadIdx.y;	//y-index for the current thread

	//exit if the output pixel is outside of the output image
	if(xi >= w - K + 1 || yi >= h - K + 1) return;

	//create registers for shared memory access
	size_t Sw = blockDim.x + K - 1;					//width of shared memory
	size_t syi = threadIdx.y;						//shared memory y-index

	//copy block and curtain data from global memory (input image) to shared memory
	for(size_t sxi = threadIdx.x; sxi < Sw; sxi += blockDim.x){
		S[(syi * Sw + sxi) * 3 + 0] = in[(yi * w + bxi + sxi) * 3 + 0];
		S[(syi * Sw + sxi) * 3 + 1] = in[(yi * w + bxi + sxi) * 3 + 1];
		S[(syi * Sw + sxi) * 3 + 2] = in[(yi * w + bxi + sxi) * 3 + 2];
	}

	__syncthreads();						//synchronize threads after the global->shared copy

	size_t i = (yi * (w - K + 1) + xi) * 3;			//1D index for the output image
	
	
	
	float3 sum;								//allocate a register to store the pixel sum
	sum.x = sum.y = sum.z = 0.0f;
	size_t in_i;
	size_t ypart = (syi * Sw + threadIdx.x) * 3;
	#pragma unroll 20
	for(size_t kxi = 0; kxi < K; kxi++){ 				//for each element in the kernel
		in_i = ((yi) * w + xi + kxi) * 3;		//calculate an index for the input image
		
		//sum.x += in[in_i + 0];						//blur each channel
		//sum.y += in[in_i + 1];
		//sum.z += in[in_i + 2];
		sum.x += S[ypart + kxi * 3 + 0];
		sum.y += S[ypart + kxi * 3 + 1];
		sum.z += S[ypart + kxi * 3 + 2];
	}
	
	out[i + 0] = sum.x / K;							//output the result for each channel
	out[i + 1] = sum.y / K;
	out[i + 2] = sum.z / K;
}

int main(int argc, char* argv[]){
	if(argc != 3){						//throw an error if the user doesn't provide an image
		std::cout<<"ERROR: you need intput"<<std::endl;
		exit(1);
	}

	stim::image<unsigned char> I(argv[1]);	//load the input image
	int K = atoi(argv[2]);					//load the kernel width (user input)

	//allocate space for the input image
	unsigned char* gpu_I;
	size_t Ibytes = I.width() * I.height() * sizeof(unsigned char) * 3;	//image size(in bytes)
	HANDLE_ERROR( cudaMalloc(&gpu_I, Ibytes) );		//allocate space on the GPU

	//copy input image to the CUDA device
	HANDLE_ERROR( cudaMemcpy(gpu_I, I.data(), Ibytes, cudaMemcpyHostToDevice));

	//calculate size of the output image
	if(K >= I.width()){
		std::cout<<"ERROR: don't be an idiot."<<std::endl;
		exit(1);
	}
	size_t w_out = I.width() - K + 1;							//width of the output image
	size_t h_out = I.height() - K + 1;							//height of the output image
	size_t Rbytes = w_out * h_out * sizeof(unsigned char) * 3;	//# of bytes in the output image
	
	//allocate space for the output image
	unsigned char* gpu_R;							//create a device pointer to R
	HANDLE_ERROR( cudaMalloc(&gpu_R, Rbytes) );		//allocate space for R
	HANDLE_ERROR( cudaMemset(gpu_R, 0, Rbytes) );	//set the initial value of R to zero

	//create a CUDA grid configuration
	dim3 threads(32, 32);							//create a square block of 1024 threads
	dim3 blocks(w_out/threads.x + 1, h_out / threads.y + 1);	//calculate # of blocks

	//calculate the required size of shared memory
	size_t Sbytes = ((threads.x + K - 1) * threads.y * sizeof(unsigned char)) * 3;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	if(props.sharedMemPerBlock < Sbytes){
		std::cout<<"ERROR: insufficient shared memory"<<std::endl;
		exit(1);
	}

	//call a CUDA kernel
	kernel<<<blocks, threads, Sbytes>>>(gpu_I, gpu_R, I.width(), I.height(), K);

	stim::image<unsigned char> R(w_out, h_out, 3);
	HANDLE_ERROR( cudaMemcpy(R.data(), gpu_R, Rbytes, cudaMemcpyDeviceToHost) );
	R.save("output.ppm");

	HANDLE_ERROR( cudaFree(gpu_I) );
	HANDLE_ERROR( cudaFree(gpu_R) );
}