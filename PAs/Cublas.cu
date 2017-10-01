#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <string>
#include <fstream>
#include <time.h>

using namespace std;
class getInput{

	unsigned int M; // rows
	unsigned int N; // columns
	float* d; // data
public:
	getInput(){ // default constructor
			M = 0;
			N = 0;
			d = NULL;

	}

	getInput(const char* filename){

		ifstream infile;												// read the input matrix file
		infile.open(filename, std::ios::in | std::ios::binary); 		// open as binary input file
		infile.read((char*)&M, sizeof(unsigned int)); 				// read the number of rows
		infile.read((char*)&N, sizeof(unsigned int)); 				// read the number of columns
		d = (float*) malloc (M * N * sizeof(float)); 						// allocating memory for the matrix
		infile.read( (char*)d , sizeof(float) * M * N); 					// read and copy the data to the memory
		infile.close();

	}
__host__ __device__ unsigned int rows(){
		return M;
	}
__host__ __device__ unsigned int cols(){
		return N;
	}
__host__ __device__ float * data(){
		return d;
	}
};

void gpu_mul(float *A, float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1.0;
	const float bet = 1.0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

int main(int argc, char *argv[]) {
cudaEvent_t start, stop;							//declaring start & stop time events
cudaEventCreate(&start);
cudaEventCreate(&stop);
	
	getInput A(argv[1]);				//calling the getInput Function to read the two array data
	getInput B(argv[2]);
	unsigned int R_A = A.rows();
	unsigned int C_A = A.cols();
	unsigned int R_B = B.rows();
	unsigned int C_B = B.cols();
	
	float *h_C = (float *)malloc(R_A* C_B * sizeof(float));			// Allocate host memory for output array C

	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,R_A * C_A * sizeof(float));
	cudaMalloc(&d_B,R_B * C_B * sizeof(float));
	cudaMalloc(&d_C,R_A * C_B * sizeof(float));

	
	cudaMemcpy(d_A,A.data(),R_A * C_A * sizeof(float),cudaMemcpyHostToDevice);		// copy the values of Array A & B to GPU
	cudaMemcpy(d_B,B.data(),R_B * C_B * sizeof(float),cudaMemcpyHostToDevice);
cudaEventRecord(start);
	
	gpu_mul(d_A, d_B, d_C, R_A, R_B, R_A);		// Multiply A and B on GPU by calling function gpu_mul using CUBLAS library 
cudaEventRecord(stop);
	
	cudaMemcpy(h_C,d_C,R_A * C_B * sizeof(float),cudaMemcpyDeviceToHost);		// Copy (and print) the result on host memory
cudaEventSynchronize(stop);
float milliseconds;
cudaEventElapsedTime(&milliseconds, start, stop);	
	ofstream outfile;
	outfile.open(argv[3], std::ios::out);
		if(outfile.is_open()){
			for (unsigned int p = 1; p < R_A+1; p++){			//priting result h_C on host to txt file
				for (unsigned int q = 1; q < C_B+1; q++){
					outfile << h_C[(p-1) * R_A + (q-1)];
					outfile << "\t";
				}
			outfile << endl;
			}
		outfile << "Total GPU time using Cublas: " << milliseconds;
		}
	
		else{
			cout << "Error in opening the file" << endl;
		}

	
	cudaFree(d_A);
	cudaFree(d_B);			//Free GPU memory
	cudaFree(d_C);	

	
	
	free(h_C);			// Free CPU memory

	return 0;
}
