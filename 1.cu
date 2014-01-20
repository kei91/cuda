/*
Написать программу, с использованием технологии CUDA, позволяющую найти 
сумму всех элементов массива из 100х100 элементов. Исходный массив необходимо 
заполнить на хосте, случайными числами от -100 до 100.
*/
#include <stdio.h>

#define M 10
#define N 100

__global__ void add (int *in, int *out) {
	
	__shared__ int data[M];
    
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	data[tid] = in[i];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
	if (tid < s) data[tid] += data[tid + s];
	__syncthreads();
	}
	if (tid == 0) out[blockIdx.x] = data[0];

}

void printError(const char *msg, const cudaError_t &error)
{
	printf("%s: ", msg);
	if (error == cudaSuccess) {
		printf("ok");
	} else {
	    printf("%s", cudaGetErrorString(error));
	}
	printf("\n");
}

int main() {

	srand(time(NULL));
	
	int *arr1 = new int[N*N];
	int *arr2 = new int[N*N];	
	int result = 0;
	for (int i = 0; i < N*N; ++i) {
		arr1[i] = rand() % N - N;
		result += arr1[i];
	}
	printf("result = %d\n", result);

	int *dev_arr1;
	int *dev_arr2;
	
	uint size = N*N*sizeof(int);
	printError("selecting device", cudaSetDevice(0));
	printError("malloc for dev_arr1", cudaMalloc((void**)&dev_arr1, size));
	printError("malloc for dev_arr2", cudaMalloc((void**)&dev_arr2, size));

	printError("memcpy dev_arr1", cudaMemcpy(dev_arr1, arr1, size, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(M, M);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	add<<<numBlocks, threadsPerBlock>>>(dev_arr1, dev_arr2);
	printError("thread sycnrhonize", cudaThreadSynchronize());

	printError("memcpy dev_arr1 back", cudaMemcpy(arr2, dev_arr1, size, cudaMemcpyDeviceToHost));

	result = 0;	
	for (int i = 0; i < N*N; ++i) {
		result += arr2[i];
	}
	printf("result = %d\n", result);
	printError("free dev_arr1", cudaFree(dev_arr1));
	printError("free dev_arr2", cudaFree(dev_arr2));
	return 0;
}
