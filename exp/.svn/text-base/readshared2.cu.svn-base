#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>

__global__ void my_kernel(uint16_t *data)
{
extern __shared__ uint16_t x[];
    x[-16] = 0x1234;
    __syncthreads();
    data[0] = x[-16]; // %gridflags
    data[1] = x[-15]; // %ntid.x 
    data[2] = x[-14]; // %ntid.y
    data[3] = x[-13]; // %ntid.z
    data[4] = x[-12]; // %nctaid.x
    data[5] = x[-11]; // %nctaid.y
    data[6] = x[-10]; // %ctaid.x
    data[7] = x[-9];  // %ctaid.y
}

int main()
{
    int width = 8;
    int size = width*4;

    uint16_t *data, *gdata;

    cudaMalloc((void**)&gdata, size);
    data = (uint16_t*)malloc(size);

    for(int x=0; x<8; ++x)
    {
        dim3 block_size(1,2,1);
        dim3 grid_size(8,8,1);
        int shared_size = 0;

        my_kernel<<<grid_size, block_size, shared_size>>>(gdata);

        cudaMemcpy((void*)data, (void*)gdata, size, cudaMemcpyDeviceToHost);

        for(int x=0; x<width; ++x)
            printf("%04x ", data[x]);
        printf("\n");
    }

    return 0;
}
