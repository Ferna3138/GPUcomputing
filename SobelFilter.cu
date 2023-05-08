#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cmath>
#include <ctime>

#include <sys/timeb.h>
#include <windows.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/video.hpp"


using namespace std;
using namespace cv;


//CUDA Implementations
__global__ void sobelFilterKernel(int w, int h, unsigned char* source, unsigned char* dest) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Boundary check
	if (x > 0 &&
		x < w - 1 &&
		y > 0 &&
		y < h - 1) {

		int gx = -source[w * (y - 1) + (x - 1)] + source[w * (y - 1) + (x + 1)] +
			(-2) * source[w * (y)+(x - 1)] + 2 * source[w * (y)+(x + 1)] +
			-source[w * (y + 1) + (x - 1)] + source[w * (y + 1) + (x + 1)];

		int gy = -source[w * (y - 1) + (x - 1)] - 2 * source[w * (y - 1) + x]
			- source[w * (y - 1) + (x + 1)] + source[w * (y + 1) + (x - 1)] + 2 * source[w * (y + 1) + x] +
			source[w * (y + 1) + (x + 1)];

			dest[w * y + x] = (int)sqrt((float)gx * (float)gx + (float) gy * (float) gy);
	}
}


__global__ void boxFilterKernel(int w, int h, unsigned char* source, unsigned char* dest, int bw, int bh) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int count = 0;
	float sum = 0.0;

	//Boundary check
	for (int j = -(bh / 2); j <= (bh / 2); j++) {
		for (int i = -(bw / 2); i <= (bw / 2); i++) {
			if ((x + i) < w &&
				(x + i) >= 0 &&
				(y + j) < h &&
				(y + j) >= 0) {
				
				sum += (float)source[((y + j) * w) + (x + i)];
				count++;
			}
		}
	}
	//Avg sum
	sum /= (float)count;
	dest[(y * w) + x] = (unsigned char) sum;
}


void boxFilter(int w, int h, unsigned char *source, unsigned char *dest, int bw, int bh) {
	//Memory allocation
	unsigned char* devSource, * devDest;

	cudaHostGetDevicePointer(&devSource, source, 0);
	cudaHostGetDevicePointer(&devDest, dest, 0);

	//Run box filter kernel
	dim3 blocks(w / 16, h / 16);
	dim3 threads(16, 16);

	boxFilterKernel << <blocks, threads >> > (w, h, devSource, devDest, bw, bh);
	cudaThreadSynchronize();
}


void sobelFilter(int w, int h, unsigned char* source, unsigned char* dest) {
	//Memory allocation
	unsigned char* devSource, * devDest;

	cudaHostGetDevicePointer(&devSource, source, 0);
	cudaHostGetDevicePointer(&devDest, dest, 0);

	//Run box filter kernel
	dim3 blocks(w / 16, h / 16);
	dim3 threads(16, 16);

	sobelFilterKernel << <blocks, threads >> > (w, h, devSource, devDest);
	cudaThreadSynchronize();

}

//Buffer

unsigned char* createImageBuffer(unsigned int bytes) {
	unsigned char* ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	return ptr;
}

void destroyImageBuffer(unsigned char* bytes) {
	cudaFreeHost(bytes);
}



int main() {
	VideoCapture camera(0);
	Mat frame;

	if (!camera.isOpened()) return -1;

	camera.set(CAP_PROP_FRAME_WIDTH, 620);
	camera.set(CAP_PROP_FRAME_HEIGHT, 480);

	camera >> frame;

	
	Mat sGray(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
	Mat dGray(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
	Mat eGray(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height));
	
	cvtColor(frame, dGray, COLOR_BGR2GRAY);
	cvtColor(frame, eGray, COLOR_BGR2GRAY);

	namedWindow("Source");
	namedWindow("Grayscale");
	namedWindow("Blurred");
	namedWindow("Sobel");

	// CUDA events 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	while (true) {

		camera >> frame;
		
		cvtColor(frame, sGray, COLOR_BGR2GRAY);


		cudaEventRecord(start);
		{
		boxFilter(frame.size().width, frame.size().height, sGray.data, dGray.data, 10, 10);
		sobelFilter(frame.size().width, frame.size().height, dGray.data, eGray.data);
		}

		cudaEventRecord(stop);

		//Display the elapsed time
		float time = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cout << "Calculated GPU time: " << time << " milliseconds" << endl;


		imshow("Source", frame);
		imshow("Grayscale", sGray);
		imshow("Blurred", dGray);
		imshow("Sobel", eGray);


		if (waitKey(30) == 27) {
			return 0;
		}
	
	}

	//Exit
	destroyImageBuffer(sGray.data);
	destroyImageBuffer(dGray.data);
	destroyImageBuffer(eGray.data);

	return 0;
}
