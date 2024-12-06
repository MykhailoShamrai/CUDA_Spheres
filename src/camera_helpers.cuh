#ifndef CAMERA_HELPERS
#define CAMERA_HELPERS

#include <cuda_runtime.h>
#include "../objects/light_sources.cuh"
#include "../objects/spheres.cuh"
#include "../external/cuda-samples/helper_math.h"

extern __host__ __device__ float angle_to_rad(float angle);
extern __host__ __device__ void rotate_positions(float* x, float* z, float* x_unrot, float* z_unrot, float angle);

extern __global__ void rotate_objects(Spheres spheres, LightSources lights, float angle_x_spheres, float angle_y_spheres,
	float angle_x_lights, float angle_y_lights, int ns, int nl);
#endif 