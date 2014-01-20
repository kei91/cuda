/*
Имеется 2 массива 100х100, первый массив заполнен нулями. Второй массив 
заполнен единицами. Написать алгоритм с использованием CUDA, позволяющий 
получить третий массив, заполненный элементами первого и второго массива, 
расположенными в шахматном порядке. Вывести на экран центральные 20х20 
элементов третьего массива.
*/

#include <stdio.h>

#define M 10
#define N 100

__global__ void add (int *arr1, int *arr2, int *arr3) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < N && y < N) {
            if ((x % 2 == 0 && y % 2 == 0) || (x % 2 != 0 && y % 2 != 0))
                    arr3[y*N + x] = arr2[y*N + x];
            if ((x % 2 == 0 && y % 2 != 0) || (x % 2 != 0 && y % 2 == 0))
                    arr3[y*N + x] = arr1[y*N + x];
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

	int *arr1 = new int[N*N];
    int *arr2 = new int[N*N];
    for (int i = 0; i < N*N; ++i) {
        arr1[i] = 0;
        arr2[i] = 1;
    }

	int *dev_arr1;
    int *dev_arr2;
    int *dev_arr3;

	uint memory_size = (N*N)*sizeof(int);
	printError("selecting device", cudaSetDevice(0));

	printError("malloc for A", cudaMalloc((void**)&dev_arr1, memory_size ));
	printError("malloc for B", cudaMalloc((void**)&dev_arr2, memory_size ));
	printError("malloc for C", cudaMalloc((void**)&dev_arr3, memory_size ));

	printError("memcpy A", cudaMemcpy(dev_arr1, arr1, memory_size , cudaMemcpyHostToDevice));
	printError("memcpy B", cudaMemcpy(dev_arr2, arr2, memory_size , cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(M, M);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	add<<<numBlocks, threadsPerBlock>>>(dev_arr1, dev_arr2, dev_arr3);
	printError("thread sycnrhonize", cudaThreadSynchronize());

	printError("memcpy C back", cudaMemcpy(arr1, dev_arr3, memory_size , cudaMemcpyDeviceToHost));

	for (int i = 40; i < 60; ++i) {
		for (int j = 40; j < 60; ++j) {
			printf("%d ",arr1[i*N + j]);
		}
		printf("\n");
	}

	printError("free A", cudaFree(dev_arr1));
	printError("free B", cudaFree(dev_arr2));
	printError("free C", cudaFree(dev_arr3));
	return 0;
}
