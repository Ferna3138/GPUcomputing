#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

__global__ void HistogramCUDA(unsigned char* image, int* histogram) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int imageIdx = x + y * gridDim.x;

	histogram[image[imageIdx]] += 1;
	//atomicAdd(&histogram[image[imageIdx], 1);
}


int main() {

	Mat InputImage = imread("Mara.jpg", 0); //Read gray scale img

	cout << "Image height : " << InputImage.rows << ", Image width: " << InputImage.cols << "Image channels: " << InputImage.channels() << endl;

	int HistogramGraySclae[256] = { 0 };


	unsigned char* devImage = NULL;
	int* devHistogram = NULL;

	//Allocate cuda variable memory
	cudaMalloc((void**)&devImage, InputImage.rows * InputImage.cols * InputImage.channels());
	cudaMalloc((void**)&devHistogram, 256 * sizeof(int));

	//Copy CPU to GPU
	cudaMemcpy(devImage, InputImage.data, InputImage.rows * InputImage.cols * InputImage.channels(), cudaMemcpyHostToDevice);
	cudaMemcpy(devHistogram, HistogramGraySclae, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 gridImage(InputImage.cols, InputImage.rows);
	HistogramCUDA << < gridImage, 1 >> > (devImage, devHistogram);

	cudaMemcpy(HistogramGraySclae, devHistogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(devHistogram);
	cudaFree(devImage);


	imwrite("Histogram_Image.jpg", InputImage);

	for (int i = 0; i < 256; i++) {
		cout << "Histogram [" << i << "]: " << HistogramGraySclae[i] << endl;
	}

	system("pause");

	return 0;
}