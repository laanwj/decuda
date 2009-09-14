#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>

__global__ void my_kernel(uint32_t *data, uint32_t p0, uint32_t p1, uint32_t p2)
{
    //data[threadIdx.x] = __mul24(threadIdx.x,p0) + __mul24(threadIdx.y,0x1234);
    data[threadIdx.x] = threadIdx.x;
}
#if 0
int main()
{
    int width = 8;
    int size = width*4;

    uint32_t *data, *gdata;

    cudaMalloc((void**)&gdata, size);
    data = (uint32_t*)malloc(size);

    dim3 block_size(8,1,1);
    dim3 grid_size(1,1,1);
    int shared_size = 0;

    my_kernel<<<grid_size, block_size, shared_size>>>(gdata, 100);

    cudaMemcpy((void*)data, (void*)gdata, size, cudaMemcpyDeviceToHost);

    for(int x=0; x<width; ++x)
        printf("%08x ", data[x]);
    printf("\n");
    return 0;
}
#endif