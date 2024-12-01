#include <cuda_runtime.h>
#include "../objects/spheres.cuh"
#include "../objects/light_sources.cuh"
#include "device_launch_parameters.h"
#include "../objects/hit_obj.cuh"
#include "renderers_helper.cuh"

__device__ void refresh_bitmap(unsigned char* bitmap, Spheres* spheres, int ns,
	LightSources* lights, int nl, int width, int heith)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= width && j >= heith)
		return;
	// here can be an error
	float ii = i - i / 2;
	float jj = -(j - j / 2);
	HitObj hit = find_intersection(ii, jj, spheres, ns);
	float3 ia = make_float3(0, 0, 0);
	// TODO: count proper color for each 
	float3 color = find_color_for_hit(hit, spheres, lights, nl, &ia, ii, jj);
	int pos = (i + width * j) * 3; // I have 3 chars for every pixel
	bitmap[pos] = color.x * 255;
	bitmap[pos + 1] = color.y * 255;
	bitmap[pos + 2] = color.z * 255;
}