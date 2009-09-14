__global__ void my_kernel(float *x)
{
	*x = max(*x, 3.14f);
}
