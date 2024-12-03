#include "camera_helpers.cuh"
#include <math.h>
#define M_PI acos(-1.0)

float angle_to_rad(int angle)
{
	return (M_PI * angle / 180);
}

float3 rotate_camera_y(int angle, float3 camera_pos)
{
	float rad_angle = angle_to_rad(angle);
	float3 first = make_float3(cos(rad_angle), 0, -sin(rad_angle));
	float3 second = make_float3(0, 1, 0);
	float3 third = make_float3(sin(rad_angle), 0, cos(rad_angle));
	return make_float3(dot(camera_pos, first), dot(camera_pos, second), dot(camera_pos, third));
}