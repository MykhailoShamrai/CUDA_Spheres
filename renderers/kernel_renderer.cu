#include <cuda_runtime.h>
#include "kernel_renderer.cuh" 
#include "device_launch_parameters.h"
#include <stdio.h>
#include "renderers_helper.cuh"

#define MAX 1000

__global__ void refresh_bitmap(unsigned char* bitmap, Spheres spheres, int ns,
	LightSources lights, int nl, int width, int heith, float3 camera_pos)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= width || j >= heith)
		return;

    extern __shared__ float array[];
	//float* x = array;
	//float* y = &x[ns];
	//float* z = &y[ns];
	//float* radius = &z[ns];

	int is = threadIdx.x + threadIdx.y * blockDim.x;
	int iss = is * 4;
	while (is < ns)
	{
		//x[is] = spheres.x[is];
		//y[is] = spheres.y[is];
		//z[is] = spheres.z[is];
		//radius[is] = spheres.radius[is];
		array[iss] = spheres.x[is];
		array[iss + 1] = spheres.y[is];
		array[iss + 2] = spheres.z[is];
		array[iss + 3] = spheres.radius[is];
		is += blockDim.x * blockDim.y;
		iss = is * 4;
	}
	__syncthreads();

	// Next idea - use atomicMin, to find the minimum in every delta

	// here can be an error
	float ii = i - width / 2;
	float jj = (j - heith / 2);
	HitObj hit = find_intersection_gpu_ver2(ii, jj, array, ns, camera_pos, is);
	float3 ia = make_float3(1, 1, 1);
	float3 color = find_color_for_hit(hit, spheres, lights, nl, &ia, ii, jj);
	int pos = (i + width * j) * 3; // I have 3 chars for every pixel
	bitmap[pos] = color.x * 255;
	bitmap[pos + 1] = color.y * 255;
	bitmap[pos + 2] = color.z * 255;
}