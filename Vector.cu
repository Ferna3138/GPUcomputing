#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < n) {
		c[tid] = a[tid] + b[tid];
	}
}


void matrix_init(int* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 100;
	}
}

void error_check(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

void init_vector(int* a, int* b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;

		printf("A: %d ", a[i]);
		printf("B: %d \n", b[i]);
	}
	
}

int main() {
	int id = cudaGetDevice(& id);
	int n = 1 << 16;

	size_t bytes = sizeof(int) * n;
	
	int* a, * b, * c;

	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	init_vector(a, b, 4);

	//Threadblock size
	int BLOCK_SIZE = 256;
	//Grid size
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);

	//CUDA Kernel
	vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>> (a, b, c, n);
	
	//Make sure all previous operations have terminated
	cudaDeviceSynchronize();

	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	error_check(a, b, c, 4);

	printf("Success! Result = ");
	for (int i = 0; i < 4; i++) {
		printf("%d, ", c[i]);
	
	}

	return 0;
}