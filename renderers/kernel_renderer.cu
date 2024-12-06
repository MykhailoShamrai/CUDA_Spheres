#include <cuda_runtime.h>
#include "kernel_renderer.cuh" 
#include <device_launch_parameters.h>
#include <stdio.h>
#include "renderers_helper.cuh"


__global__ void refresh_bitmap(float* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith, float3 camera_pos)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= width || j >= heith)
		return;

	// The idea is to find on the start of each frame generation, which sferes can probably be visible on the block's part of the screen
	// Find the corners to bound block's part of the screen
	int x_min = blockIdx.x * blockDim.x - width / 2;
	int y_max = -(blockIdx.y * blockDim.y - heith / 2);
	int x_max = blockDim.x + blockIdx.x * blockDim.x - width / 2;
	int y_min = -(blockDim.y + blockIdx.y * blockDim.y - heith / 2);

	// Now each thread must take one sphere and specify if it has chance to be present in this part of a screen;
	// For that purpose I need to have some dinamically alocated structure, like list or I don't know
	int is = threadIdx.x + threadIdx.y * blockDim.x;
	extern __shared__ unsigned char array[];
	while (is < ns)
	{
		check_if_sphere_is_visible_for_block(x_min, y_max, x_max, y_min, spheres.x[is],
			spheres.y[is], spheres.z[is], spheres.radius[is], array, is, camera_pos);
		is += blockDim.x * blockDim.y;
	}
	__syncthreads();

	int ii = i - width / 2;
	int jj = -(j - heith / 2);
	//printf("dupa\n");
	HitObj hit = find_intersection_gpu_ver3(ii, jj, spheres, array, ns, camera_pos, is);
	float3 color = find_color_for_hit(hit, spheres, lights, nl, ii, jj, camera_pos);
	int pos = (i + width * j) * 3;
	bitmap[pos] = color.x;
	bitmap[pos + 1] = color.y;
	bitmap[pos + 2] = color.z;
}
