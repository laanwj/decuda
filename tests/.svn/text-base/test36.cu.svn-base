__constant__ int uu;
__constant__ short vv;
__constant__ int ww[10];
__device__ int xx;
__device__ int *xx2;

__global__ void f_test(unsigned int *data, unsigned char *data2)
{
	for(unsigned int x=0; x<10; ++x)
	{
		unsigned int y = data[x-2];
		data2[x] = data[x];	
		data2[x+2] = y>>16;
	}
	data[0] = uu;
	data[1] = vv;
	for(unsigned int x=0; x<10; ++x)
	{
            data[x] = ww[x];
	}
	xx = uu;
	xx2 = &xx;
}
