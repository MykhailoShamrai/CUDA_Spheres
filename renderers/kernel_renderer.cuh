#pragma once

#include <cuda_runtime.h>
#include "../objects/light_sources.cuh"
#include "../objects/spheres.cuh"
#include "../objects/hit_obj.cuh"

__global__ void refresh_bitmap(unsigned char* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith);