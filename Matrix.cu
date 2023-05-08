#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiledMatrixMult(int* a, int* b, int* c, int n, int tileSize) {
	//Two statically sized pieces of shared mem
	__shared__ int A[SHMEM_SIZE];
	__shared__ int B[SHMEM_SIZE];

	//Shorten these params for clean re-use
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	//Global coordinates
	int row = by * tileSize + ty;
	int col = bx * tileSize + tx;

	//Intermediate sum for elements
	int tempSum = 0;

	//Sweep tiles over entire matrix
	//we use total size / tile size which is the amount of tile steps
	for (int i = 0; i < (n / tileSize); i++) {
		A[(ty * tileSize) + tx] = a[row * n + (i * tileSize + tx)];
		B[(tx * tileSize) + tx] = b[(i * tileSize * n + ty *n ) + col];
		
		__syncthreads();

		for (int j = 0; j < tileSize; j++) {
			tempSum += A[(ty * tileSize) + j] * B[(j * tileSize) + tx];
		}
	
		__syncthreads();
	}
	c[(row * n) + col] = tempSum;
}

void init_Matrix(int *a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = rand() % 10;
		}
	}
}

void check_answer(int* a, int* b, int* c, int n) {
	int* verifyC;
	verifyC = (int*)malloc(n * n * sizeof(int));
	int tempVal;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			tempVal = 0;
			for (int k = 0; k < n; k++) {
				tempVal += a[i * n + k] * b[k * n * j];
			}
			verifyC[i * n + j] = tempVal;
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			assert(c[i * n + j] == verifyC[i * n + j]);
		}
	}
}

int main() {
	//1024 x 1024 matrix
	int n = 1 << 10;

	size_t bytes = n * n * sizeof(int);

	int* ha, * hb, * hc;
	int* da, * db, * dc;

	ha = (int*)malloc(bytes);
	hb = (int*)malloc(bytes);
	hc = (int*)malloc(bytes);

	cudaMalloc(&da, bytes);
	cudaMalloc(&db, bytes);
	cudaMalloc(&dc, bytes);

	init_Matrix(ha, n);
	init_Matrix(hb, n);

	cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
	
	int BLOCK_SIZE = 16;

	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	tiledMatrixMult << <grid, threads >> > (da, db, dc, n, BLOCK_SIZE);

	cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);

	//check_answer(ha, hb, hc, n);

	free(ha);
	free(hb);
	free(hc);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	printf("Completed successfully\n");

	return 0;
}