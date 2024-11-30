// Structure that represents position of hit of ray and sphere. Also contains index of a sphere that ray hits.
#include "spheres.cuh"
#include <cuda_runtime.h>
struct HitObj
{
	float x;
	float y;
	float z;
	int index;
};

__host__ __device__ HitObj find_intersection(float ray_x, float ray_y, Spheres* spheres, int n);