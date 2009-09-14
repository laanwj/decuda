#include <cuda_runtime_api.h>
#include <cuda.h>
#include <algorithm>
#include <cassert>

#define SAFE_CALL(x) assert((x)==CUDA_SUCCESS);

int main()
{
    CUmodule m = 0;
    CUdevice dev = 0;
    CUcontext ctx = 0;
    
    /// Init cuda
    SAFE_CALL(cuInit(0));
    /// Get device
    SAFE_CALL(cuDeviceGet(&dev, 0));
    /// Create context
    SAFE_CALL(cuCtxCreate(&ctx, 0, dev));
    
    SAFE_CALL(cuModuleLoad(&m, "test.cubin"));

    CUfunction hfunc;
    SAFE_CALL(cuModuleGetFunction(&hfunc, m, "my_kernel"));
    
    
    int width = 16;
    int size = width*4;

    uint32_t *data=0;
    CUdeviceptr gdata;
    
    SAFE_CALL(cuMemAlloc(&gdata, size));
    SAFE_CALL(cuMemsetD32(gdata, 0, size/4));

    data = (uint32_t*)malloc(size);

    SAFE_CALL(cuFuncSetBlockShape(hfunc, 16, 1, 1));
    SAFE_CALL(cuFuncSetSharedSize(hfunc, 0));
    SAFE_CALL(cuParamSetSize(hfunc, 8)); // 8+4
    SAFE_CALL(cuParamSetv(hfunc, 0, &gdata, sizeof(gdata)));
    //SAFE_CALL(cuParamSeti(hfunc, 4, 1));
    //SAFE_CALL(cuParamSeti(hfunc, 8, 2));
    //SAFE_CALL(cuParamSeti(hfunc, 12, 3));
    //SAFE_CALL(cuParamSeti(hfunc, 16, 0));
    //SAFE_CALL(cuParamSeti(hfunc, 20, 0));
    int shared_size = 0;
    
    SAFE_CALL(cuLaunch(hfunc));
    //SAFE_CALL(cuLaunchGrid(hfunc, 3, 4));

    SAFE_CALL(cuMemcpyDtoH((void*)data, gdata, size));

    for(int x=0; x<width; ++x)
        printf("%08x ", data[x]);
    printf("\n");
    for(int x=0; x<width; ++x)
        printf("%f ", *(float*)(&data[x]));
    printf("\n");

    return 0;
}
