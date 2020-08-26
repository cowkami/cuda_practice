#include <stdio.h>

#define N 10000000000

__global__ void add_gpu(float *a, float *b, float *c) {
	long long tid = blockIdx.x;
	if (tid < N) 
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	float a[N], b[N], c[N];
	float *dev_a, *dev_b, *dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));
	cudaMalloc((void**)&dev_c, N * sizeof(float));

	for (long long i=0; i<N; i++) {
		a[i] = -i;
		b[i] = i*3;
	}

	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	add_gpu<<<N, 1>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	// for (int i=0; i<N; i++) {
	// 	printf("%d + %d = %d\n", a[i], b[i], c[i]);
	// }

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
