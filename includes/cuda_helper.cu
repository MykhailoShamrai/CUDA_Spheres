#include "cuda_helper.cuh"
#include <stdlib.h>

float rand_float(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

