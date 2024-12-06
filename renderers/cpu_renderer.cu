#define SIZE_OF_BLOCK 64
#include "renderers_helper.cuh"

void refresh_bitmap_cpu(float* bitmap, Spheres spheres,
	int ns, LightSources lights, int nl, int width, int heith, float3 camera_pos)
{

	// Firstly I'll find spheres that are interesting for blocks of a screen
	int block_x = (width + SIZE_OF_BLOCK - 1) / SIZE_OF_BLOCK;
	int block_y = (heith + SIZE_OF_BLOCK - 1) / SIZE_OF_BLOCK;


	// Allocate memory for vectors
	int num_blocks = block_x * block_y;
	std::vector<std::vector<int>> v (num_blocks);


	for (int bj = 0; bj < block_y; bj++)
	{
		for (int bi = 0; bi < block_x; bi++)
		{
			int ind_x_min = bi * SIZE_OF_BLOCK;
			int ind_x_max = SIZE_OF_BLOCK + bi * SIZE_OF_BLOCK;
			int ind_y_max = SIZE_OF_BLOCK + bj * SIZE_OF_BLOCK;
			int ind_y_min = bj * SIZE_OF_BLOCK;


			int x_min = ind_x_min - width / 2;
			int y_max = -(ind_y_min - heith / 2);
			int x_max = ind_x_max - width / 2;
			int y_min = -(ind_y_max - heith / 2);
			for (int k = 0; k < ns; k++)
			{
				check_if_sphere_is_visible_for_block_cpu(x_min, y_max, x_max, y_min, spheres.x[k], spheres.y[k],
					spheres.z[k], spheres.radius[k], k, camera_pos, &v[bi + bj * block_x]);
			}
			//if (v[bi + bj * block_x].size() > 0)
			{
				for (int j = ind_y_min; j < ind_y_max && j < heith; j++)
				{
					for (int i = ind_x_min; i < ind_x_max && i < width; i++)
					{			
						int ii = i - width / 2;
						int jj = -(j - heith / 2);
						HitObj hit = find_intersection_cpu(ii, jj, spheres, ns, camera_pos, v[bi + bj * block_x]);
						float3 color = make_float3(0, 0, 0);
						if (v[bi + bj * block_x].size() > 0)
							color = find_color_for_hit(hit, spheres, lights, nl, ii, jj, camera_pos);
						int pos = (i + width * j) * 3; // I have 3 chars for every pixel
						//printf("pos %d, %d, %d\n", pos, pos + 1, pos + 2);

						bitmap[pos] = color.x;
						bitmap[pos + 1] = color.y;
						bitmap[pos + 2] = color.z;
					}
				}
			}
		}
	}
}
