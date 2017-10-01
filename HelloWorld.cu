#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void add (int *aa, int *bb, int *cc)
{
	int tid = threadIdx.x;
	if(tid<10)
	cc[tid]=aa[tid]+bb[tid];
}

int main()
{

int a[10];
int b[10];
int c[10];

for(int i=0; i<10; i++)
{
a[i]=i+9;
b[i]=a[i]+3;
c[i]=0
}

int *g_a;
int *g_b;
int *g_c;

int size=sizeof(int)*10;

cudaMalloc((void**)&g_a, size);
cudaMalloc((void**)&g_b, size);
cudaMalloc((void**)&g_c, size);

cudaMemcpy(g_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(g_b, b, size, cudaMemcpyHostToDevice);


add<<<1,10>>>(g_a,g_b,g_c);

cudaMemcpy(c, g_c, size, cudaMemcpyDeviceToHost);

for(int i=0; i<10;i++)
	cout<<a[i]<<" + "<<b[i]<<" = "<<c[i]<<"\n";

	return 0;

}