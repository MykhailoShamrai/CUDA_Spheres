#include <cuda_runtime.h>
#include "../objects/spheres.cuh"
#include "../objects/light_sources.cuh"
#include "../includes/cuda_helper"

//__device__ refresh_bitmap(unsigned char* bitmap, Spheres* spheres, int ns,
//	LightSources* lights, int nl, float3 hit_pos, )