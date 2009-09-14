#define TYPE char
__global__ void f_test(TYPE *data, TYPE *data2)
{
__shared__ TYPE x[256];
__shared__ TYPE y;
	x[0] = *data;
	x[1] = *data;
	x[2] = *data;
	x[3] = *data;
	__syncthreads();
	data2[0] = x[0];
	data2[1] = x[1];
	data2[2] = x[2];
	data2[3] = x[3];
	data2[4] = x[4];
	data2[5] = x[5];
	data2[6] = x[6];
	data2[7] = x[7];
 	data2[8] = y;


}
