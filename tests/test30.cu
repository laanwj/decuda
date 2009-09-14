
__global__ void bra_test(short *data)
{
    data[0] = blockIdx.x;
    data[1] = blockIdx.y;
    data[2] = blockIdx.z;
    data[3] = threadIdx.x;
    data[4] = threadIdx.y;
    data[5] = threadIdx.z;
    data[6] = blockDim.x;
    data[7] = blockDim.y;
    data[8] = blockDim.z;
    data[9] = gridDim.x;
    data[10] = gridDim.y;
    data[11] = gridDim.z;
}
