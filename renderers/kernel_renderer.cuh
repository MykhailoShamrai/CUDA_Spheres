#pragma once

#include <cuda_runtime.h>
#include "../objects/light_sources.cuh"
#include "../objects/spheres.cuh"
#include "../objects/hit_obj.cuh"
#include "renderers_helper.cuh"

__global__ void refresh_bitmap(unsigned char* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith);
__host__ __device__ HitObj find_intersection(float ray_x, float ray_y, Spheres spheres, int n);



__host__ __device__ float3 find_color_for_hit(HitObj hit, Spheres spheres, LightSources lights, int nl, float3* ia, int i, int j);