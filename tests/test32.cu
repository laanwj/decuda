
__global__ void f_test(float *data, float *data2)
{
	*data = ceil(*data2);
}
