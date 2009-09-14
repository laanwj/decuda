#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>


__global__ void my_kernel(uint32_t *data)
{
extern __shared__ int x[];
    data[0] = x[-8];
    data[1] = x[-7];
    data[2] = x[-6];
    data[3] = x[-5];
    data[4] = blockDim.x;
    data[5] = blockDim.y;
}

int main()
{
    int width = 6;
    int size = width*4;

    uint32_t *data, *gdata;

    cudaMalloc((void**)&gdata, size);
    data = (uint32_t*)malloc(size);

    for(int x=0; x<8; ++x)
    {
        dim3 block_size;
        dim3 grid_size;
        int shared_size;

        block_size.x = 1;
        block_size.y = 2;
        block_size.z = 1;
        grid_size.x = 8;
        grid_size.y = 8;
        grid_size.z = 1;
        shared_size = 0;

        my_kernel<<<grid_size, block_size, shared_size>>>(gdata);

        cudaMemcpy((void*)data, (void*)gdata, size, cudaMemcpyDeviceToHost);

        for(int x=0; x<width; ++x)
        {
            printf("%08x ", data[x]);
        }
        printf("\n");
    }

    return 0;

}
