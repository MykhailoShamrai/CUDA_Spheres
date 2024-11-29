#include "sphere.cuh"
#include <stdlib.h>
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

	for (int i = 0; i < n; i++)
	{
		// Random axis for a sphere
		*(spheres->x) = rand_float(0, 500);
		*(spheres->y) = rand_float(0, 500);
		*(spheres->z) = rand_float(0, 500);
	
		*(spheres->radius) = rand_float(2, 10);
	}
	return;
}

void h_clean_memory_for_spheres(Spheres* spheres, int n)
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