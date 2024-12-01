#include "renderers_helper.cuh"
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

__host__ __device__ float3 find_color_for_hit(HitObj hit, Spheres* spheres, LightSources* lights, int nl, float3* ia, int i, int j)
{
	// If no sphere intersection is detected
	if (hit.index == -1)
		return make_float3(1, 1, 1);
	float3 observer_pos = make_float3(i, j, 0);
	float3 sphere_center = make_float3(spheres->x[hit.index], spheres->y[hit.index], spheres->z[hit.index]);
	float3 sphere_color = make_float3(spheres->R[hit.index], spheres->G[hit.index], spheres->B[hit.index]);
	float3 hit_pos = make_float3(hit.x, hit.y, hit.z);
	// Find normal
	float3 N = normalize(hit_pos - sphere_center);
	// Find vector to observer
	float3 V = normalize(observer_pos - hit_pos);
	// For each Light Source find vector to light
	float3 light_pos = make_float3(0, 0, 0);
	float3 light_color;
	float3 L;

	float3 R;
	float LN_dot_prod;
	float RV_dot_prod;
	float3 color_of_pixel = make_float3(0, 0, 0);
	for (int k = 0; k < nl; k++)
	{
		light_pos = make_float3(lights->x[k], lights->y[k], lights->z[k]);
		light_color = make_float3(lights->R[k], lights->G[k], lights->B[k]);

		L = normalize(L - hit_pos);

		// Also here find R vector
		R = normalize(2 * dot(L, N) * N - L);

		LN_dot_prod = dot(L, N);
		RV_dot_prod = dot(R, V);

		LN_dot_prod = LN_dot_prod >= 0 ? LN_dot_prod : 0;
		RV_dot_prod = RV_dot_prod >= 0 ? RV_dot_prod : 0;

		color_of_pixel += spheres->kd[hit.index] * LN_dot_prod * sphere_color + spheres->ks[hit.index] * pow(RV_dot_prod, spheres->alpha[hit.index]) * light_color;
	}

	color_of_pixel += spheres->ka[hit.index] * (*ia);
	color_of_pixel = clamp(color_of_pixel, make_float3(0, 0, 0), make_float3(1, 1, 1));
	return color_of_pixel;
}