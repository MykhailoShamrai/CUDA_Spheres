#include "spheres.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../includes/cuda_helper.cuh"

void h_allocate_memory_for_spheres(Spheres* spheres, int n)
{
	spheres->x_unrotated = (float*)malloc(sizeof(float) * n);
	spheres->y_unrotated = (float*)malloc(sizeof(float) * n);
	spheres->z_unrotated = (float*)malloc(sizeof(float) * n);
	spheres->x = (float*)malloc(sizeof(float) * n);
	spheres->y = (float*)malloc(sizeof(float) * n);
	spheres->z = (float*)malloc(sizeof(float) * n);
	spheres->radius = (float*)malloc(sizeof(float) * n);
	spheres->R = (float*)malloc(sizeof(float) * n);
	spheres->G = (float*)malloc(sizeof(float) * n);
	spheres->B = (float*)malloc(sizeof(float) * n);
	spheres->ka = (float*)malloc(sizeof(float) * n);
	spheres->kd = (float*)malloc(sizeof(float) * n);
	spheres->ks = (float*)malloc(sizeof(float) * n);
	spheres->alpha = (float*)malloc(sizeof(float) * n);
}

void h_clean_memory_for_spheres(Spheres* spheres)
{
	free(spheres->x_unrotated);
	free(spheres->y_unrotated);
	free(spheres->z_unrotated);
	free(spheres->x);
	free(spheres->y);
	free(spheres->z);
	free(spheres->radius);
	free(spheres->R); 
	free(spheres->G); 
	free(spheres->B); 
	free(spheres->ka);
	free(spheres->kd);
	free(spheres->ks);
	free(spheres->alpha);
}

void d_allocate_memory_for_spheres(Spheres* spheres, int n)
{
	checkCudaErrors(cudaMalloc((void**)&spheres->x_unrotated, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&spheres->y_unrotated, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&spheres->z_unrotated, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->x), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->y), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->z), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->radius), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->R), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->G), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->B), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->ka), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->kd), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->ks), sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&(spheres->alpha), sizeof(float) * n));
}

void d_clean_memory_for_spheres(Spheres* spheres)
{
	checkCudaErrors(cudaFree(spheres->x_unrotated));
	checkCudaErrors(cudaFree(spheres->y_unrotated));
	checkCudaErrors(cudaFree(spheres->z_unrotated));
	checkCudaErrors(cudaFree(spheres->x));
	checkCudaErrors(cudaFree(spheres->y));
	checkCudaErrors(cudaFree(spheres->z));
	checkCudaErrors(cudaFree(spheres->radius));
	checkCudaErrors(cudaFree(spheres->R));
	checkCudaErrors(cudaFree(spheres->G));
	checkCudaErrors(cudaFree(spheres->B));
	checkCudaErrors(cudaFree(spheres->ka));
	checkCudaErrors(cudaFree(spheres->kd));
	checkCudaErrors(cudaFree(spheres->ks));
	checkCudaErrors(cudaFree(spheres->alpha));
}

void create_random_spheres(Spheres* spheres, int n)
{
	for (int i = 0; i < n; i++)
	{
		spheres->x[i] = rand_float(-1000, 1000);
		spheres->y[i] = rand_float(-1000, 1000);
		spheres->z[i] = rand_float(-1000, 1000);
		spheres->x_unrotated[i] = spheres->x[i];
		spheres->y_unrotated[i] = spheres->y[i];
		spheres->z_unrotated[i] = spheres->z[i];
		spheres->radius[i] = rand_float(10, 20);
		spheres->R[i] = rand_float(0, 1);
		spheres->G[i] = rand_float(0, 1);
		spheres->B[i] = rand_float(0, 1);
		spheres->ka[i] = rand_float(0, 0.2);
		spheres->kd[i] = rand_float(0, 0.5);
		spheres->ks[i] = rand_float(0, 1);
		spheres->alpha[i] = 100;
	}
}