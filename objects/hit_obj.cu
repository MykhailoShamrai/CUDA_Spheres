#include "hit_obj.cuh"
#include "../external/cuda-samples/helper_math.h"
#include <cuda_runtime.h>
#include <float.h>

__host__ __device__ HitObj find_intersection(float ray_x, float ray_y, Spheres* spheres, int n)
{
	HitObj res;
	res.x = 0;
	res.y = 0;
	res.z = FLT_MAX;
	res.index = -1;
	float ray_z = -1;
	float radius;
	float3 A = make_float3(ray_x, ray_y, 0);
	float3 B = make_float3(0, 0, 1);
	float3 C;
	// I have unit vector for B, so a is 1
	float a = 1.0f;
	float b;
	float c;
	float d;
	float step1;
	float step2;
	for (int i = 0; i < n; i++)
	{
		C = make_float3(spheres->x[i], spheres->y[i], spheres->z[i]);
		radius = spheres->radius[i];
		b = 2 * dot(B, A - C);
		c = dot(A - C, A - C) - radius * radius;
		d = b * b - 4 * c;
		if (d >= 0)
		{
			step1 = (-b - sqrt(d)) / 2;
			step2 = (-b + sqrt(d)) / 2;
			// whole wphere is behind camera
			if (step1 >= 0 && step2 >= 0 && step1 < res.z)
			{
				res.z = step1;
				res.index = i;	
			}
			// camera is inside a sphere
			else if (step1 < 0 && step2 >= 0)
			{
				res.z = FLT_MAX;
				res.index = -1;
				break;
			}
		}
	}
	return res;
}