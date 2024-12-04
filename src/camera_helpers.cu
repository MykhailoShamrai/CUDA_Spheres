#include "camera_helpers.cuh"
#include <math.h>
#define M_PI acos(-1.0)

float angle_to_rad(float angle)
{
	return (M_PI * angle / 180);
}

float3 rotate_camera_y(int angle, float3 camera_pos)
{
	float rad_angle = angle_to_rad(angle);
	float3 first = make_float3(cos(rad_angle), 0, -sin(rad_angle));
	float3 second = make_float3(0, 1, 0);
	float3 third = make_float3(sin(rad_angle), 0, cos(rad_angle));
	return make_float3(cuda_examples::dot(camera_pos, first), cuda_examples::dot(camera_pos, second), cuda_examples::dot(camera_pos, third));
}

void rotate_positions(float* x, float* z, float* x_unrot, float* z_unrot, float angle, int n)
{
	float rad_angle = angle_to_rad(angle);
	float cos_rad = cos(rad_angle);
	float sin_rad = sin(rad_angle);
	float3 third = make_float3(sin(rad_angle), 0, cos(rad_angle));
	for (int i = 0; i < n; i++)
	{
		x[i] = cos_rad * x_unrot[i] + (-sin_rad * z_unrot[i]);
		z[i] = sin_rad * x_unrot[i] + cos_rad * z_unrot[i];
	}
}
