#pragma once

struct LightSources
{
	float* x_unrotated;
	float* y_unrotated;
	float* z_unrotated;

	float* x;
	float* y;
	float* z;

	float* R;
	float* G;
	float* B;
};

void h_allocate_memory_for_light_sources(LightSources* lights, int n);
void h_clean_memory_for_light_sources(LightSources* lights);
void d_allocate_memory_for_light_sources(LightSources* lights, int n);
void d_clean_memory_for_light_sources(LightSources*);
void create_random_light_sources(LightSources* sources, int n);
