#include <cuda_runtime.h>
#include <stdlib.h>
#include "../includes/cuda_helper.cuh"
#include "light_sources.cuh"


void h_allocate_memory_for_light_sources(LightSources* lights, int n)
{
	lights->x_unrotated = (float*)malloc(sizeof(float) * n);
	lights->y_unrotated = (float*)malloc(sizeof(float) * n);
	lights->z_unrotated = (float*)malloc(sizeof(float) * n);
	lights->x = (float*)malloc(sizeof(float) * n);
	lights->y = (float*)malloc(sizeof(float) * n);
	lights->z = (float*)malloc(sizeof(float) * n);
	lights->R = (float*)malloc(sizeof(float) * n);
	lights->G = (float*)malloc(sizeof(float) * n);
	lights->B = (float*)malloc(sizeof(float) * n);
}


void h_clean_memory_for_light_sources(LightSources* lights)
{
	free(lights->x_unrotated);
	free(lights->y_unrotated);
	free(lights->z_unrotated);
	free(lights->x);
	free(lights->y);
	free(lights->z);
	free(lights->R);
	free(lights->G);
	free(lights->B);
}

void d_allocate_memory_for_light_sources(LightSources* lights, int n)
{
	checkCudaErrors(cudaMalloc((void**)&(lights->x_unrotated), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->y_unrotated), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->z_unrotated), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->x), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->y), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->z), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->R), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->G), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(lights->B), sizeof(float) * n));
}

void d_clean_memory_for_light_sources(LightSources* lights)
{
	checkCudaErrors(cudaFree(lights->x_unrotated));
	checkCudaErrors(cudaFree(lights->y_unrotated));
	checkCudaErrors(cudaFree(lights->z_unrotated));
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
		lights->x[i] = rand_float(-1000, 1000);
		lights->y[i] = rand_float(-1000, 1000);
		lights->z[i] = rand_float(-1000, 1000);
		lights->x_unrotated[i] = lights->x[i];
		lights->y_unrotated[i] = lights->y[i];
		lights->z_unrotated[i] = lights->z[i];
		lights->R[i] = rand_float(0, 1);
		lights->G[i] = rand_float(0, 1);
		lights->B[i] = rand_float(0, 1);
	}
}