#pragma once 
#include "../objects/light_sources.cuh"
#include "../objects/spheres.cuh"
#include "../objects/hit_obj.cuh"

void refresh_bitmap_cpu(float* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith, float3 camera_pos);