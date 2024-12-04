#pragma once

#include <cuda_runtime.h>
#include "../objects/light_sources.cuh"
#include "../objects/spheres.cuh"
#include "../objects/hit_obj.cuh"

__global__ void refresh_bitmap(unsigned char* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith, float3 camera_pos);

__global__ void refresh_bitmap_ver2(unsigned char* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith, float3 camera_pos);