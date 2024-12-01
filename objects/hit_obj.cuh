// Structure that represents position of hit of ray and sphere. Also contains index of a sphere that ray hits.
#pragma once

#include "spheres.cuh"
#include <cuda_runtime.h>
struct HitObj
{
	float x;
	float y;
	float z;
	int index;
};
