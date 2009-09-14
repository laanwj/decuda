__global__ void my_kernel(float *x, char *y)
{
char a,b,c,d,e,f,g;
	a = y[0];
	b = y[1];
	c = y[2];
	d = y[3];
	e = y[4];
	f = y[5];
	g = y[6];
	x[0] = a;
	x[1] = b;
	x[2] = c;
	x[3] = d;
	x[4] = e;
	x[5] = f;
	x[6] = g;
}
