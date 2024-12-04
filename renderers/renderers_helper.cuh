#ifndef RENDER_GPU_H
#define RENDER_GPU_H

#include <cuda_runtime.h>  
#include "../objects/hit_obj.cuh"
#include <../external/cuda-samples/helper_math.h>
#include "../objects/light_sources.cuh"
#include <float.h>
#include <stdio.h>

// extern __device__ HitObj find_intersection_gpu_ver2(float ray_x, float ray_y, float* arr, int n, float3 camera_pos, int num);
extern __host__ __device__ HitObj find_intersection(float ray_x, float ray_y, Spheres spheres, int n, float3 camera_pos);
extern __host__ __device__ float3 find_color_for_hit(HitObj hit, Spheres spheres, LightSources lights, int nl, float3* ia, int i, int j);
// extern __device__ HitObj find_intersection_gpu(float ray_x, float ray_y, float* x, float* y, float* z, float* radiuses, int n, float3 camera_pos);
extern __device__ HitObj find_intersection_gpu_ver3(float ray_x, float ray_y, Spheres spheres, unsigned char* array, int n, float3 camera_pos, int num);
extern __device__ void check_if_sphere_is_visible_for_block_ver2(int x_min, int y_max, int x_max, int y_min, float x, 
	float y, float z, float radius, unsigned char* array, int index, float3 camera_pos);
extern __device__ void check_if_sphere_is_visible_for_block(int x_min, int y_max, int x_max, int y_min, float x, float y, float z, float radius,
	unsigned char* array, int index, float3 camera_pos);

#endif