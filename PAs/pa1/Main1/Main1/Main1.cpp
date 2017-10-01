// Main1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<iostream>
#include<fstream>
#include<string>
using namespace std;


void read_file(string filename);

int main()
{

	std::string input_filename;
	cin >> input_filename;
	read_file(input_filename);
	return 0;
}
void read_file(string filename)
{

	ifstream file(filename, std::ios::in | std::ios::binary);
	if (!file) {
		std::cout << "Error opening input file in image::load_netpbm()" << std::endl;
		exit(1);
	}
	if (file.is_open())
	{
		cout << "Name of the file:\t" << filename << '\n';

		char format[2];												//allocate space to hold the image format tag
		file.read(format, 2);										//read the image format tag
		file.seekg(1, std::ios::cur);								//skip the newline character

		if (format[0] == 'P' && format[1] == '6')
		{
			cout << "Correct format image file:\t" << format[0] << format[1] << endl;

		}
		else
		{
			cout << "Error in image::load_netpbm() - file format tag is invalid: " << format[0] << format[1] << std::endl;
			exit(1);
		}


		unsigned char c;								//stores a character
		while (file.peek() == '#') 
		{					//if the next character indicates the start of a comment
			while (true) 
			{
				c = file.get();
				if (c == 0x0A) break;
			}
		}
		std::string sw;									//create a string to store the width of the image
		while (true) 
		{
			c = file.get();							//get a single character
			if (c == ' ') break;						//exit if we've encountered a space
			sw.push_back(c);							//push the character on to the string
		}
		size_t w = atoi(sw.c_str());					//convert the string into an integer
		cout << "Width of Image:\t" << w << endl;
		int row = w;

		std::string sh;
		while (true) 
		{
			c = file.get();
			if (c == 0x0A) break;
			sh.push_back(c);
		}

		size_t h = atoi(sh.c_str());					//convert the string into an integer
		cout << "Height of Image: " << h << endl;
		int col = h;

		std::string sints;
		while (true) 
		{
			c = file.get();
			if (c == 0x0A) break;
			sints.push_back(c);
		}

	}
}

