#ifndef CAMERA_HELPERS
#define CAMERA_HELPERS

#include <cuda_runtime.h>
#include "../objects/light_sources.cuh"
#include "../objects/spheres.cuh"
#include "../external/cuda-samples/helper_math.h"

//extern __global__ rotate_spheres(Spheres spheres, )
extern float angle_to_rad(int angle);
extern float3 rotate_camera_y(int angle, float3 camera_pos);

#endif 