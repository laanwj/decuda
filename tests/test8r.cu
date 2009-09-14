__global__ void my_kernel(int *x, int *y, int *z)
{
	*x = __mul24((*y), *z);
}
