#define TYPE char
__global__ void f_test(TYPE data3, int *data, TYPE *data2, TYPE data4)
{
__shared__ TYPE y[256];
__shared__ TYPE x;
	y[100] = y[101];
	x = *data;
	__syncthreads();
	data2[0] = x;
	data2[1] = x;
	data2[2] = data3;
	data2[3] = data4;
}
