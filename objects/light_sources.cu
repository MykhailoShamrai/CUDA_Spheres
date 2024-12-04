#include <cuda_runtime.h>
#include <stdlib.h>
#include "../includes/cuda_helper.cuh"
#include "light_sources.cuh"


void h_allocate_memory_for_light_sources(LightSources* lights, int n)
{
	lights->x = (float*)malloc(sizeof(float) * n);
	lights->y = (float*)malloc(sizeof(float) * n);
	lights->z = (float*)malloc(sizeof(float) * n);
	lights->R = (float*)malloc(sizeof(float) * n);
	lights->G = (float*)malloc(sizeof(float) * n);
	lights->B = (float*)malloc(sizeof(float) * n);
}


void h_clean_memory_for_light_sources(LightSources* lights)
{
	free(lights->x);
	free(lights->y);
	free(lights->z);
	free(lights->R);
	free(lights->G);
	free(lights->B);
}

void d_allocate_memory_for_light_sources(LightSources* lights, int n)
{
	checkCudaErrors(cudaMalloc((void**)&(lights->x), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->y), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->z), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->R), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->G), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->B), sizeof(float) * n));
}

void d_clean_memory_for_light_sources(LightSources* lights)
{
	checkCudaErrors(cudaFree(lights->x));
	checkCudaErrors(cudaFree(lights->y));
	checkCudaErrors(cudaFree(lights->z));
	checkCudaErrors(cudaFree(lights->R));
	checkCudaErrors(cudaFree(lights->G));
	checkCudaErrors(cudaFree(lights->B));
}

void create_random_light_sources(LightSources* lights, int n)
{
	for (int i = 0; i < n; i++)
	{
		lights->x[i] = rand_float(-500, 500);
		lights->y[i] = rand_float(-500, 500);
		lights->z[i] = rand_float(-500, 500);
		lights->R[i] = 1; //rand_float(0, 1);
		lights->G[i] = 1; //rand_float(0, 1);
		lights->B[i] = 1; //rand_float(0, 1);
		//lights->x[i] = 100;
		//lights->y[i] = 100;
		//lights->z[i] = 100;
		//lights->R[i] = 1;
		//lights->G[i] = 1;
		//lights->B[i] = 1;
	}
}