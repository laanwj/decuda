__global__ void my_kernel(float *x)
{
	*x = min(*x, 3.14f);
}
