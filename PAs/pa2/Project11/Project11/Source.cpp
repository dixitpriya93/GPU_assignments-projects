#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char **argv)
{
	unsigned int R;
	unsigned int C;
	string filename (argv[1]);
	ifstream iA;
	iA.open(filename, ios::in | ios::binary);
	if (iA.is_open()) {
		iA.read((char*)&R, sizeof(unsigned int));
		cout << R << endl;
		iA.read((char*)&C, sizeof(unsigned int));
		cout << C << endl;
		float *A = (float*)malloc(R * C * sizeof(float));
		iA.read((char*)A, sizeof(float) * R * C);
		iA.close();
	}
	else { cout << "Error in opening file" << endl; }
	return 0;
}
