#define n_threads_per_block  256

__global__ static void multiply64(unsigned long long *d)
  {
    d[threadIdx.x + n_threads_per_block * blockIdx.x] += 1023237287261818171ull;
    d[threadIdx.x + n_threads_per_block * blockIdx.x] *= 1905125066741695459ull;
  }
