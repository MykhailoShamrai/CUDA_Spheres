#include <cuda_runtime.h>
#include "kernel_renderer.cuh" 
#include "device_launch_parameters.h"
#include <stdio.h>
#include "renderers_helper.cuh"

__global__ void refresh_bitmap(unsigned char* bitmap, Spheres spheres, int ns,
	LightSources lights, int nl, int width, int heith, float3 camera_pos)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= width || j >= heith)
		return;
	// here can be an error
	float ii = i - width / 2;
	float jj = (j - heith / 2);
	HitObj hit = find_intersection(ii, jj, spheres, ns, camera_pos);
	float3 ia = make_float3(1, 1, 1);
	float3 color = find_color_for_hit(hit, spheres, lights, nl, &ia, ii, jj);
	int pos = (i + width * j) * 3; // I have 3 chars for every pixel
	bitmap[pos] = color.x * 255;
	bitmap[pos + 1] = color.y * 255;
	bitmap[pos + 2] = color.z * 255;
}