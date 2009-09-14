__global__ void my_kernel(int *out, int width)
{
    extern __shared__ int sha[];
    int x=3;

    while(x<width/2)
    {
        sha[x] = 200;
        out[x] = sha[x+1];
	x = out[x+1];
    }
    __syncthreads();
    while(x<width/2)
    {
        sha[x] = 200;
        out[x] = sha[x+1];
	x = out[x+1];
    }
}
