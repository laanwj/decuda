__global__ void my_kernel(int *x, int *y)
{
	*x = (*y) ^ 3;
}
