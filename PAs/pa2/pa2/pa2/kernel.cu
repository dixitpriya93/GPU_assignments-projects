#define USING_OPENCV

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <fstream>
#include <ctime>


using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;


void handleError(cudaError_t error, const char* text) {

	if (error != cudaSuccess) {
		cout << "ERROR " << text << endl;
		exit(1);
	}
}

class loadPropsFromBinaryFile {

	unsigned int r;					// rows
	unsigned int c;					// columns
	float* d;						// data

public:

	loadPropsFromBinaryFile() {                                       // default constructor 
		r = 0;
		c = 0;
		d = NULL;
	}

	loadPropsFromBinaryFile(std::string filename) {

		ifstream infile;												// to read the matrix 

		infile.open(filename, std::ios::in | std::ios::binary);		// open as binary input file

		if (infile.fail()) {											// check for error in opening the file
			cerr << "ERROR opening the first file" << endl;
			exit(1);
		}
		infile.read((char*)&r, sizeof(unsigned int));				// assuming the first value is number of rows
		infile.read((char*)&c, sizeof(unsigned int));				// assuming the second value is the number of columns

		d = (float*)malloc(r * c * sizeof(float));				// allocate memory for the matrix
		infile.read((char*)d, sizeof(float) * r * c);			// read and copy the data to the memory

		infile.close();
	}

	__host__ __device__	unsigned int rows() {                                      // return number of rows
		return r;
	}

	__host__ __device__	unsigned int cols() {                                       // return number of columns
		return c;
	}

	__host__ __device__ float* data() {                                            // return the pointer to matrix data
		return d;
	}

};
__global__ void kernelMatMultiplication(loadPropsFromBinaryFile matA, loadPropsFromBinaryFile matB, float* matA_data, float* matB_data, float* matC_data) {

	size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;					// index to the threads across x dimension
	size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;					// index to the threads across y dimension

	if (idx_x > matB.cols() || idx_y > matA.rows()) return;

	extern __shared__ float sharedBytes[];

	float* sharedA = sharedBytes;
	float* sharedB = sharedBytes + blockDim.y * matA.cols() * sizeof(float);
	float* sharedC = sharedB + blockDim.x * matB.rows() * sizeof(float);

	//if(idx_x == 0 && idx_y == 0)
	//printf("%f		%f\n", sharedBytes[]);


}
float* hostMatMultiplication(loadPropsFromBinaryFile a, loadPropsFromBinaryFile b) {

	float* c = (float*)malloc(a.rows() * b.cols() * sizeof(float));												// allocate memory for A * B

	float temp = 0;
	for (int i = 0; i < a.rows(); i++) {																			 // using matrix A as reference to assign the indices
		for (int rowIdxA = 0; rowIdxA < a.rows(); rowIdxA++) {
			for (int colIdxA = 0; colIdxA < a.cols(); colIdxA++)
				temp += a.data()[colIdxA * a.rows() + rowIdxA] * b.data()[colIdxA + a.cols() * i];		 // store the dot product in a register

			c[i * a.rows() + rowIdxA] = temp;																		 // store in heap
			temp = 0;
		}
	}

	return c;
}

int main(int argc, char *argv[]) {

	loadPropsFromBinaryFile A(argv[1]);						// read binary file A
	loadPropsFromBinaryFile B(argv[2]);						// read binary file B

	if (A.cols() != B.rows()) {                              // check if the right marices are passed into the main function
		cerr << "ERROR: the dimensions of the two matrices don't agree for multiplication" << endl;
		exit(1);
	}


	// profiling
	float* C = NULL;										// pointer to mutliplication result matrix on cpu
	C = hostMatMultiplication(A, B);							// implementation of matrix multiplication on cpu
																// profiling


	cudaError_t error;
	cudaDeviceProp prop;								// for the device properties

	error = cudaGetDeviceProperties(&prop, 0);						// getting the device properties
	handleError(error, "getting the device properties");


	// allocating memory for matrix A on the device
	float* device_A;
	size_t bytesA = A.rows() * A.cols() * sizeof(float);
	error = cudaMalloc((void**)&device_A, bytesA);
	handleError(error, "allocating memory on the device for matrix A");


	// allocating memory for matrix B on the device
	float* device_B;
	size_t bytesB = B.rows() * B.cols() * sizeof(float);
	error = cudaMalloc((void**)&device_B, bytesB);
	handleError(error, "allocating memory on the device for matrix B");


	// allocating memory for matrix A * B = C on the device
	float* device_C;
	size_t bytesC = A.cols() * B.rows() * sizeof(float);
	error = cudaMalloc((void**)&device_C, bytesC);
	handleError(error, "allocating memory on the device for matrix C");


	// copying matrix A from host to device
	error = cudaMemcpy(device_A, A.data(), bytesA, cudaMemcpyHostToDevice);
	handleError(error, "copying matrix A from host to device");


	// copying matrix B from host to device
	error = cudaMemcpy(device_B, B.data(), bytesB, cudaMemcpyHostToDevice);
	handleError(error, "copying matrix B from host to device");


	dim3 threads(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));				// two dimensional block configuration
	dim3 blocks(ceil(B.cols() / threads.x), ceil(A.rows() / threads.y));							// two dimensional grid configuration

	size_t sharedBytesPerBlock_A = threads.y * A.cols() * sizeof(float);
	size_t sharedBytesPerBlock_B = threads.x * B.rows() * sizeof(float);
	size_t sharedBytesPerBlock_C = threads.x * threads.y * sizeof(float);
	size_t sharedBytesTotal = sharedBytesPerBlock_A + sharedBytesPerBlock_B + sharedBytesPerBlock_C;

	if (sharedBytesTotal > prop.sharedMemPerBlock) {
		cout << "ERROR: insufficient shared memory" << endl;
		exit(1);
	}

	kernelMatMultiplication << <blocks, threads, sharedBytesTotal >> >(A, B, device_A, device_B, device_C);

	//cudaThreadSynchronize();


	//cudaFree(device_A);

	return 0;
}
