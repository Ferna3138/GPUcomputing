#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <stdlib.h>

using namespace std;

//convolutional mask = 7x7
#define MASK_DIM 3

#define MASK_OFFSET (MASK_DIM/2)

//Allocate mask in constant memory
__constant__ int mask[7 * 7];


__global__ void convolution_2d(int* matrix, int* result, int N) {
	//Calculate the global thread positions
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//starting index for calculation
	int startRow = row - MASK_OFFSET;
	int startCol = col - MASK_OFFSET;

	//Accumulating result
	int temp = 0;

	//Iterate over rows
	for (int i = 0; i < MASK_DIM; i++) {
		//Iterate over cols
		for (int j = 0; j < MASK_DIM; j++) {
			//Range check rows
			if ((startRow + i) >= 0 && (startRow + i) < N) {
				//Range check cols
				if ((startCol + j) >= 0 && (startCol + j) < N) {

					//Accumulate result
					temp += matrix[(startRow + i) * N + (startCol + j)] *
						mask[i * MASK_DIM + j];

				}
			}
		}
	}
	//Write result
	result[row * N + col] = temp;
}


void initMatrix(int* a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = rand() % 10;
			printf("%d", a[i * n + j]);
		}
		printf("\n");
	}
}

void initConstMatrix(int* a, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i * n + j] = 1;
			printf("%d", a[i * n + j]);
		}
		printf("\n");
	}
}


void verifyResult(int* m, int* mask, int* result, int N) {
	int temp;

	int offsetRow;
	int offsetCol;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			temp = 0;

			for (int k = 0; k < MASK_DIM; k++) {
				offsetRow = i - MASK_OFFSET + k;

				for (int l = 0; l < MASK_DIM; l++) {
					offsetCol = j - MASK_OFFSET + l;

					if (offsetRow >= 0 && offsetRow < N) {
						if (offsetCol >= 0 && offsetCol < N) {
							temp += m[offsetRow * N + offsetCol] * mask[k * MASK_DIM + l];
						}
					}
				}
			}
			assert(result[i * N + j] == temp);
		}
	}
	printf("Result is correct!\n");
}

unsigned char* createImageBuffer(unsigned int bytes, unsigned char** devicePtr)
{
	unsigned char* ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

int main() {
	//1024 x 1024
	int N = 1 << 4;

	size_t bytes_n = N * N * sizeof(int);

	//Allocate matrix and initialisation
	int* matrix = new int[N * N];
	int* result = new int[N * N];
	printf("Original matrix: \n");
	initMatrix(matrix, N);

	printf("-------------------------------- \n");

	//Size of the mask in bytes
	size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

	//Allocate the mask and initialisation
	int* h_mask = new int[MASK_DIM * MASK_DIM];
	printf("Mask: \n");
	initMatrix(h_mask, MASK_DIM);
	
	//Allocate device memory
	int* d_matrix;
	int* d_result;
	cudaMalloc(&d_matrix, bytes_n);
	cudaMalloc(&d_result, bytes_n);

	//Copy data to device
	cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(mask, h_mask, bytes_m);

	//Calculate grid dimensions
	int THREADS = 16;
	int BLOCKS = (N + THREADS - 1) / THREADS;

	//Dimension launch arguments
	dim3 block_dim(THREADS, THREADS);
	dim3 grid_dim(BLOCKS, BLOCKS);

	printf("Result: \n");

	//Kernel
	convolution_2d << <grid_dim, block_dim >> > (d_matrix, d_result, N);

	//Copy result
	cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d", result[i * N + j]);
		}
		printf("\n");
	}

	//Verify
	verifyResult(matrix, h_mask, result, N);
	printf("\n");

	printf("Completed!\n");

	return 0;
}