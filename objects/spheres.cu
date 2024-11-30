#include "spheres.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../includes/cuda_helper.cuh"

void h_allocate_memory_for_spheres(Spheres* spheres, int n)
{
	spheres->x = (float*)malloc(sizeof(float)* n);
	spheres->y = (float*)malloc(sizeof(float)* n);
	spheres->z = (float*)malloc(sizeof(float)* n);
	spheres->radius = (float*)malloc(sizeof(float)* n);
	spheres->R = (float*)malloc(sizeof(float)* n);
	spheres->G = (float*)malloc(sizeof(float)* n);
	spheres->B = (float*)malloc(sizeof(float)* n);
	spheres->ka = (float*)malloc(sizeof(float)* n);
	spheres->kd = (float*)malloc(sizeof(float)* n);
	spheres->ks = (float*)malloc(sizeof(float)* n);
	spheres->alpha = (float*)malloc(sizeof(float)* n);
}

void h_clean_memory_for_spheres(Spheres* spheres)
{
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
		spheres->x[i] = rand_float(10, 700);
		spheres->y[i] = rand_float(10, 700);
		spheres->z[i] = rand_float(10, 700);
		spheres->radius[i] = rand_float(2, 10);
		spheres->R[i] = rand_float(0, 1);
		spheres->G[i] = rand_float(0, 1);
		spheres->B[i] = rand_float(0, 1);
		spheres->ka[i] = rand_float(0, 1);
		spheres->kd[i] = rand_float(0, 1);
		spheres->ks[i] = rand_float(0, 1);
		spheres->alpha[i] = rand_float(0, 1);
	}
}

void create_test_spheres(Spheres* spheres)
{
	spheres->x[0] = 300;
	spheres->y[0] = 500;
	spheres->z[0] = 0;
	spheres->radius[0] = 10;
	spheres->R[0] = 1;
	spheres->G[0] = 0;
	spheres->B[0] = 0;
	spheres->ka[0] = 1;
	spheres->kd[0] = 0.25;
	spheres->ks[0] = 0.25;
	spheres->alpha[0] = 1;

	spheres->x[1] = -300;
	spheres->y[1] = 500;
	spheres->z[1] = 10;
	spheres->radius[1] = 10;
	spheres->R[1] = 0;
	spheres->G[1] = 1;
	spheres->B[1] = 0;
	spheres->ka[1] = 1;
	spheres->kd[1] = 0.25;
	spheres->ks[1] = 0.25;
	spheres->alpha[1] = 1;

	spheres->x[2] = 300;
	spheres->y[2] = 500;
	spheres->z[2] = 210;
	spheres->radius[2] = 10;
	spheres->R[2] = 1;
	spheres->G[2] = 0;
	spheres->B[2] = 1;
	spheres->ka[2] = 1;
	spheres->kd[2] = 0.25;
	spheres->ks[2] = 0.25;
	spheres->alpha[2] = 1;
}

