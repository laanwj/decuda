__global__ void my_kernel(int *x, int *y)
{
	*x = min((*y), 3);
}
