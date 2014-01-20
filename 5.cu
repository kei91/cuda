/*
Написать алгоритм вычитающий 2 двумерных массива (один из другого), с 
количеством элементов 200x350 на GPU, с использованием CUDA. Элементы первого
массива задать равными (i * j), элементы второго массива вычислить по формуле 
((200 - i) * (350 - j)), где i и j — индексы элемента. После сложения вывести верхний 
левый угол результирующей матрицы (10х10 элементов), и нижний правый угол 
результирующей матрицы (10х10 элементов).
*/
#include <stdio.h>

#define M 10
#define N1 200
#define N2 350

__global__ void add (int *arr1, int *arr2, int *arr3) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N1 && y < N2) {
           arr3[y*N1 + x] = arr2[y*N1 + x] - arr1[y*N1 + x];
    }
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

	int *arr1 = new int[N1*N2];
    int *arr2 = new int[N1*N2];
    for (int i = 0; i < N1; ++i) {
    	for (int j = 0; j < N2; ++j) {
    	    arr1[i*N1 + j] = i*j;
        	arr2[i*N1 + j] = (200 - i) * (350 - j);	
    	}
    }

	int *dev_arr1;
    int *dev_arr2;
    int *dev_arr3;

	uint memory_size = (N1*N2)*sizeof(int);
	printError("selecting device", cudaSetDevice(0));

	printError("malloc for A", cudaMalloc((void**)&dev_arr1, memory_size ));
	printError("malloc for B", cudaMalloc((void**)&dev_arr2, memory_size ));
	printError("malloc for C", cudaMalloc((void**)&dev_arr3, memory_size ));

	printError("memcpy A", cudaMemcpy(dev_arr1, arr1, memory_size , cudaMemcpyHostToDevice));
	printError("memcpy B", cudaMemcpy(dev_arr2, arr2, memory_size , cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(M, M);
	dim3 numBlocks(N1 / threadsPerBlock.x, N2 / threadsPerBlock.y);
	add<<<numBlocks, threadsPerBlock>>>(dev_arr1, dev_arr2, dev_arr3);
	printError("thread sycnrhonize", cudaThreadSynchronize());

	printError("memcpy C back", cudaMemcpy(arr1, dev_arr3, memory_size , cudaMemcpyDeviceToHost));

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 10; ++j) {
			printf("%d ",arr1[i*N1 + j]);
		}
		printf("\n");
	}
	printf("------------------------------\n");
	for (int i = N1 - 10; i < N1; ++i) {
		for (int j = N2 - 10; j < N2; ++j) {
			printf("%d ",arr1[i*N1 + j]);
		}
		printf("\n");
	}

	printError("free A", cudaFree(dev_arr1));
	printError("free B", cudaFree(dev_arr2));
	printError("free C", cudaFree(dev_arr3));
	return 0;
}
