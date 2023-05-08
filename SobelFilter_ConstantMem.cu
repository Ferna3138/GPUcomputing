#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// Constant memory space to store different kernels
__constant__ float constKernel[256];

/**
 * Convolution function for cuda.  Destination is expected to have the same width/height as source, but there will be a border
 * of floor(kWidth/2) pixels left and right and floor(kHeight/2) pixels top and bottom
 *
 * @param source      Source image host pinned memory pointer
 * @param width       Source image width
 * @param height      Source image height
 * @param paddingX    source image padding along x
 * @param paddingY    source image padding along y
 * @param kOffset     offset into kernel store constant memory
 * @param kWidth      kernel width
 * @param kHeight     kernel height
 * @param destination Destination image host pinned memory pointer
 */

//CUDA Kernels
__global__ void convolve(unsigned char* source, int width, int height, int paddingX, int paddingY, size_t kOffset, int kWidth, int kHeight, unsigned char* destination) {
    int col = (blockIdx.y * blockDim.y) + threadIdx.y;
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;

    float temp= 0.0;
    int   pWidth = kWidth / 2;
    int   pHeight = kHeight / 2;

    // Only execute for valid pixels
    if (row >= pWidth + paddingX &&
        col >= pHeight + paddingY &&
        row < (blockDim.x * gridDim.x) - pWidth - paddingX &&
        col < (blockDim.y * gridDim.y) - pHeight - paddingY){

        for (int j = -pHeight; j <= pHeight; j++) {
            for (int i = -pWidth; i <= pWidth; i++) {
                // Sample the weight for this location
                int ki = (i + pWidth);
                int kj = (j + pHeight);
                float w = constKernel[(kj * kWidth) + ki + kOffset];


                temp += w * float(source[((col + j) * width) + (row + i)]);
            }
        }
    }

    destination[(col * width) + row] = (unsigned char)temp;
}

// converts the pythagoran theorem along a vector on the GPU
__global__ void pythagoras(unsigned char* a, unsigned char* b, unsigned char* c)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(a[idx]);
    float bf = float(b[idx]);

    c[idx] = (unsigned char)sqrtf(af * af + bf * bf);
}

// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
unsigned char* createImageBuffer(unsigned int bytes, unsigned char** devicePtr)
{
    unsigned char* ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}



int main(int argc, char** argv) {

    VideoCapture camera(0);
    Mat frame;
    if (!camera.isOpened())
        return -1;

    // Create windows
    namedWindow("Source");
    namedWindow("Greyscale");
    namedWindow("GaussianBlur");
    namedWindow("Sobel");

    // CUDA events 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const float gaussianKernel5x5[25] = {
        2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
        4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
        5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
        4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
        2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
    };

    const float boxBlurKernel[9] = {
        1.f, 1.f, 1.f,
        1.f, 1.f, 1.f,
        1.f, 1.f, 1.f,
    };

    cudaMemcpyToSymbol(constKernel, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);
   
    const size_t gaussianKernel5x5Offset = 0;

    // Sobel kernels on X and Y
    const float sobelKernelX[9] = {
        -1.f, 0.f, 1.f,
        -2.f, 0.f, 2.f,
        -1.f, 0.f, 1.f,
    };
    const float sobelKernelY[9] = {
        1.f, 2.f, 1.f,
        0.f, 0.f, 0.f,
        -1.f, -2.f, -1.f,
    };

    //Copy sobel kernels to constant memory adding the corresponding offsets
    cudaMemcpyToSymbol(constKernel, sobelKernelX, sizeof(sobelKernelX), sizeof(gaussianKernel5x5));
    cudaMemcpyToSymbol(constKernel, sobelKernelY, sizeof(sobelKernelY), sizeof(gaussianKernel5x5) + sizeof(sobelKernelX));

    const size_t sobelKernelXOffset = sizeof(gaussianKernel5x5) / sizeof(float);
    const size_t sobelKernelYOffset = sizeof(sobelKernelX) / sizeof(float) + sobelKernelXOffset;

    // Create CPU/GPU shared images - one for the initial and one for the result
    camera >> frame;
    unsigned char* sourceDataDevice, * blurredDataDevice, * edgesDataDevice;
    
    Mat source(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
    Mat blurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
    Mat sobel(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice));

    // Create two temporary images (for holding sobel gradients)
    unsigned char* deviceGradientX, * deviceGradientY;
    cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
    cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);


    while (true){
        // Capture the image and store a gray conversion to the gpu
        camera >> frame;
        cvtColor(frame, source, COLOR_BGR2GRAY);

        cudaEventRecord(start);
        {
            // convolution kernel launch parameters
            dim3 convBlocks(frame.size().width / 16, frame.size().height / 16);
            dim3 convThreads(16, 16);

            // pythagoran kernel launch paramters
            dim3 pblocks(frame.size().width * frame.size().height / 256);
            dim3 pthreads(256, 1);

            // Gaussian Blur
            convolve << <convBlocks, convThreads >> > (sourceDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernel5x5Offset, 5, 5, blurredDataDevice);

            // Sobel convolution (x&y padding is now 2 because there is a border of 2 around a 5x5 gaussian filtered image)
            convolve << <convBlocks, convThreads >> > (blurredDataDevice, frame.size().width, frame.size().height, 2, 2, sobelKernelXOffset, 3, 3, deviceGradientX);
            convolve << <convBlocks, convThreads >> > (blurredDataDevice, frame.size().width, frame.size().height, 2, 2, sobelKernelYOffset, 3, 3, deviceGradientY);
            pythagoras << <pblocks, pthreads >> > (deviceGradientX, deviceGradientY, edgesDataDevice);

            cudaThreadSynchronize();

        }
        cudaEventRecord(stop);

        // Display the elapsed time
        float time = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cout << "Calculated GPU time: " << time << " milliseconds" << endl;

        
        imshow("Source", frame);
        imshow("Greyscale", source);
        imshow("GaussianBlur", blurred);
        imshow("Sobel", sobel);

        if (cv::waitKey(1) == 27) break;
    }

    cudaFreeHost(source.data);
    cudaFreeHost(blurred.data);
    cudaFreeHost(sobel.data);

    cudaFree(deviceGradientX);
    cudaFree(deviceGradientY);

    return 0;
}