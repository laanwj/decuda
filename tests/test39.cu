__global__ void my_kernel(int *out, int width)
{
    if(width&3)
    {
        for(int x=0; x<width/2; ++x)
            out[x] = 123;
    }
    else
    {
        short *out2 = (short*)out;
        for(int x=0; x<width; ++x)
            out2[x] = 123;
    }
}
