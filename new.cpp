
__global__ void gpu_mul(getInput A,getInput B, float* d_A, float* d_B, float* d_C) {

	__shared__ float input_a[32];   						// Tile size of 32x32 
    	__shared__ float input_b[32];

    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float temp = 0.0;
    input_a[threadIdx.x] = 0.0;
    input_b[threadIdx.x] = 0.0;

    for (int k = 0; k < (((A.cols() - 1)/ 32) + 1); k++)
    {
        if ( (Row < A.rows()) && (threadIdx.x + (k*32)) < A.cols())
        {
            input_a[threadIdx.y][threadIdx.x] = d_A[(Row * A.cols()) + threadIdx.x + (k*32)];
        }
        else
        {
            input_a[threadIdx.y][threadIdx.x] = 0.0;
        }            
        if ( Col < B.cols() && (threadIdx.y + k*32) < B.rows())
        {
            input_b[threadIdx.y][threadIdx.x] = d_B[(threadIdx.y + k*32) * B.cols() + Col];
        }
        else
        {
            input_b[threadIdx.y][threadIdx.x] = 0.0;
        }            
        __syncthreads();

        for (int j = 0; j < 32; ++j)
        {
            temp += input_a[threadIdx.y][j] * input_b[j][threadIdx.x];
        }
    }
    if (Row < A.rows() && Col < B.cols())
    {
        d_C[Row*B.cols() + Col] = temp;
    }
}


int main(int argc, char *argv[]) {
cudaEvent_t start, stop;							//declaring start & stop time events
cudaEventCreate(&start);
cudaEventCreate(&stop);

	getInput A(argv[1]);						// reading input arrays A & B from Command Line
	getInput B(argv[2]);
float *h_C = (float *)malloc(A.rows() * B.cols() * sizeof(float));	// Allocate host memory for output matrix C						

	if (A.cols() != B.rows()) {                              		// #rows in A = #cols in B for matrix multiplication
		cerr << "Invaid Matrix Multiplication" << endl;
		exit(1);
	}
unsigned int M = A.rows();
unsigned int N = B.cols();
float* h_A = A.data();
float* h_B = B.data();

float* C = (float*) malloc (M * N * sizeof(float));
	for(unsigned int i = 1 ; i < M+1 ; i++){                             // using matrix A as reference to assign the indices
		for(unsigned int j = 1 ; j < N+1 ; j++){
			for(unsigned int k = 1 ; k < M+1 ; k++){
				C[i + M * (j-1)] +=   h_A[i + M * (k-1)]   *    h_B[k + (j-1) * N];
			}
		cout << C[i + M * (j-1)] << "  ";
		}
		cout << endl;
	}

	
cudaError_t error;
												
	float* d_A;
	size_t bytesA = A.rows() * A.cols() * sizeof(float);					// allocating GPU memory for input matrix A
	error = cudaMalloc((void**)&d_A, bytesA);
	handleError(error, "allocating GPU memory for input matrix A");


	float* d_B;
	size_t bytesB = B.rows() * B.cols() * sizeof(float);					// allocating GPU memory for input matrix B
	error = cudaMalloc((void**)&d_B, bytesB);
	handleError(error, "allocating GPU memory for matrix B");


	float* d_C;
	size_t bytesC = A.rows() * B.cols() * sizeof(float);					// allocating GPU memory for product C = A*B
	error = cudaMalloc((void**)&d_C, bytesC);
	handleError(error, "allocating memory on the device for matrix C");


	error = cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice);
	handleError(error, "copying matrix A from host to device");				// copying matrix A from host to device


	
	error = cudaMemcpy(d_B, B.data(), bytesB, cudaMemcpyHostToDevice);
	handleError(error, "copying matrix B from host to device");				// copying matrix B from host to device


	dim3 gridDim(A.rows()/32 +1, B.cols()/32 +1, 1);
	dim3 blockDim(32,32,1);					// two dimensional grid & block configuration
	
cudaEventRecord(start);							//start recording time

	gpu_mul << <gridDim, blockDim>> >(A, B, d_A, d_B, d_C);
cudaThreadSynchronize();
cudaEventRecord(stop);							//stop recording time
	
	cudaMemcpy(h_C,d_C,A.rows() * B.cols() * sizeof(float),cudaMemcpyDeviceToHost);		// Copy (and print) the result on host memory

	
cudaEventSynchronize(stop);
float milliseconds;								//time in milliseconds
cudaEventElapsedTime(&milliseconds, start, stop);

	ofstream outfile;
	outfile.open(argv[3], std::ios::out);
		if(outfile.is_open()){
			for (unsigned int p = 1; p < A.rows()+1; p++){			//priting result h_C on host to txt file
				for (unsigned int q = 1; q < B.cols()+1; q++){
					outfile << h_C[(p-1) * A.rows() + (q-1)];
					outfile << "\t";
				}
			outfile << endl;
			}
		outfile << "Total GPU time using shared memory implementation: " << milliseconds;
		}
	
		else{
			cout << "Error in opening the file" << endl;}
 
const char* file = "mse.txt";		
ofstream outfile1;
outfile1.open(file, std::ios::out);
if (outfile1.is_open()){
 for (int i=0; i < A.rows()*B.cols(); i++)
    {
        if (C[i]  != d_C[i] )
        {
            outfile1 << "Mismatch at Row = " <<  i /B.cols() << "Col = " << i % B.cols() << "cpu[] = " << C[i] << "gpu[] = " <<  d_C[i] << "\n";
	    outfile1 << endl;
            break;
        }
    }
}
else{
	cout << "Error in opening the file" << endl;
	}


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}