/*
*pa2.cpp
* Created by: Priya Dixit
*07-Nov-2016
*11:49:28 PM
*/
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char *argv[])
{

	unsigned int rows = 0;
	unsigned int cols = 0;

	float* A = (float*) malloc (rows * cols * sizeof(float));			// allocate memory for the matrixA
	float* B = (float*) malloc (rows * cols * sizeof(float));			// allocate memory for the matrixB


	ifstream infile(argv[1]);												// to read the matrix

	infile.open(infile(argv[1]), std::ios::in | std::ios::binary);		// open as binary input file

	if(infile.is_open()){											// check for error in opening the file

	infile.read((char*)&rows, sizeof(unsigned int));				// assuming the first value is number of rows
	infile.read((char*)&cols, sizeof(unsigned int));				// assuming the second value is the number of columns


	infile.read( (char*)A , sizeof(float) * rows * cols);			// read and copy the data to the memory

	}
	else { cout << "Error in opening first file" << endl; }
	infile.close();

	ifstream in;												// to read the matrix

	in.open(argv[2], std::ios::in | std::ios::binary);		// open as binary input file

	if(in.is_open()){											// check for error in opening the file

	in.read((char*)&rows, sizeof(unsigned int));				// assuming the first value is number of rows
	in.read((char*)&cols, sizeof(unsigned int));				// assuming the second value is the number of columns

	in.read( (char*)B , sizeof(float) * rows * cols);			// read and copy the data to the memory

	}
	else { cout << "Error in opening second file" << endl; }
	in.close();

return 0;
}
