#include "camera_helpers.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#define M_PI acos(-1.0)

__host__ __device__ float angle_to_rad(float angle)
{
	return (M_PI * angle / 180);
}


__host__ __device__ void rotate_positions(float* x, float* z, float* x_unrot, float* z_unrot, float angle)
{
	float rad_angle = angle_to_rad(angle);
	float cos_rad = cos(rad_angle);
	float sin_rad = sin(rad_angle);
	*x = cos_rad * *x_unrot + -sin_rad * *z_unrot;
	*z = sin_rad * *x_unrot + cos_rad * *z_unrot;
}

__global__ void rotate_objects(Spheres spheres, LightSources lights, float angle_x_spheres, float angle_y_spheres, float angle_x_lights, 
	float angle_y_lights, int ns, int nl)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < ns)
	{
		rotate_positions(&(spheres.x[i]), &(spheres.z[i]), &(spheres.x_unrotated[i]), &(spheres.z_unrotated[i]), angle_x_spheres);
		rotate_positions(&(spheres.y[i]), &(spheres.z[i]), &(spheres.y_unrotated[i]), &(spheres.z[i]), angle_y_spheres);
	}

	// I assume that we have more spheres than lights
	if (i < nl)
	{
		rotate_positions(&(lights.x[i]), &(lights.z[i]), &(lights.x_unrotated[i]), &(lights.z_unrotated[i]), angle_x_lights);
		rotate_positions(&(lights.y[i]), &(lights.z[i]), &(lights.y_unrotated[i]), &(lights.z[i]), angle_y_lights);
	}
}
