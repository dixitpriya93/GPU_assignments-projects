/*
*pa22.cpp
* Created by: Priya Dixit
*16-Nov-2016
*5:22:16 PM
*/
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;
unsigned int M;
unsigned int N;
double* A = (double*) malloc (M * N * sizeof(double));
double* B = (double*) malloc (M * N * sizeof(double));

int main(int argc, char*argv[])
{
	ifstream inputA;
		inputA.open(argv[1],std::ios::in | std::ios::binary);

		if(inputA.is_open()){

			inputA.read((char*)&M, sizeof(int));
			cout << M << endl;
			inputA.read((char*)&N, sizeof(int));
			cout << N << endl;
			double* A = (double*) malloc (M * N * sizeof(double));				// allocate memory for the matrix
			inputA.read( (char*)A , sizeof(double) * M * N);
			inputA.close();
		}
		else{
			cout << "Error in opening the file" << endl;
		}

		ifstream inputB;
				inputB.open(argv[2],std::ios::in | std::ios::binary);

				if(inputB.is_open()){

					inputB.read((char*)&M, sizeof(int));
					cout << M << endl;
					inputB.read((char*)&N, sizeof(int));
					cout << N << endl;
					double* B = (double*) malloc (M * N * sizeof(double));				// allocate memory for the matrix
					inputB.read( (char*)B , sizeof(double) * M * N);
					inputB.close();
				}
				else{
					cout << "Error in opening the file" << endl;
				}
				double temp = 0;
				double* C = (double*) malloc (M * N * sizeof(double));

					double start_time = std::clock();
					for(unsigned int i = 1 ; i < M+1 ; i++){                             // using matrix A as reference to assign the indices
						for(unsigned int j = 1 ; j < N+1 ; j++){
							for(unsigned int k = 1 ; k < M+1 ; k++){
								   temp +=   A[i + M * (k-1)]   *    B[k + (j-1) * N];
							}
							C[i + M * (j-1)] = temp;
							cout << C[i + M * (j-1)] << "  ";
							temp = 0;

						}
					 cout << endl;
				}
			ofstream outfile;
			outfile.open(argv[3], std::ios::out);
			if(outfile.is_open()){
				for (unsigned int p = 1; p < M+1; p++){
					for (unsigned int q = 1; q < M+1; q++){
						outfile << C[p + M * (q-1)];
						outfile << " ";
					}
				outfile << endl;
				}
			}
			else{
			cout << "Error in opening the file" << endl;
			}

					double end_time = std::clock();
					cout << "total time on CPU " << (end_time - start_time)/((double) CLOCKS_PER_SEC) << "  seconds" << endl;

return 0;
}
