#define TYPE char
__global__ void f_test(TYPE *data, TYPE *data2)
{
__shared__ TYPE w[256];
__shared__ TYPE x[256];
	for(int y=0; y<256; ++y)
		w[y] = 0;
	for(int y=0; y<256; ++y)
		x[y] = 0;
	__syncthreads();
	for(int y=0; y<256; ++y)
		data2[y] = x[y];


}
