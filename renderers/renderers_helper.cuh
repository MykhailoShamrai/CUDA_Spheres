#include <cuda_runtime.h>  
#include "../objects/hit_obj.cuh"
#include "../external/cuda-samples/helper_math.h"
#include "../objects/light_sources.cuh"

__host__ __device__ HitObj find_intersection(float ray_x, float ray_y, Spheres* spheres, int n);
__host__ __device__ float3 find_color_for_hit(HitObj hit, Spheres* spheres, LightSources* lights, int nl, float3 *ia, int i, int j);
