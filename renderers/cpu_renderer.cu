//#include "kernel_renderer.cuh"
#include "renderers_helper.cuh"


void refresh_bitmap_cpu(unsigned char* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith, float3 camera_pos)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < heith; j++)
		{	
			float ii = i - width / 2;
			float jj = (j - heith / 2);
			HitObj hit = find_intersection(ii, jj, spheres, ns, camera_pos);
			float3 ia = make_float3(1, 1, 1);
			// TODO: count proper color for each 
			float3 color = find_color_for_hit(hit, spheres, lights, nl, &ia, ii, jj, camera_pos);
			int pos = (i + width * j) * 3; // I have 3 chars for every pixel
			bitmap[pos] = color.x * 255;
			bitmap[pos + 1] = color.y * 255;
			bitmap[pos + 2] = color.z * 255;
		}
	}
}
