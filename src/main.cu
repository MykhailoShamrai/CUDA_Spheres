
#include <GLFW/glfw3.h>
//#include "../objects/spheres.cuh"
#include "../objects/light_sources.cuh"
#include "../includes/cuda_helper.cuh"
#include "../objects/hit_obj.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>

//#include "cmake-test-cuda.h"

#define NUMBER_OF_SPHERES 10
#define NUMBER_OF_LIGHTS 2
#define WIDTH 1500
#define HEIGHT 1000


int main(void)
{
    Spheres sphere;
    Spheres d_spheres;
    LightSources lights;
    LightSources d_lights;
    h_allocate_memory_for_spheres(&sphere, 3);
    create_test_spheres(&sphere);
    //create_random_spheres(&sphere, NUMBER_OF_SPHERES);
    HitObj obj = find_intersection(300.f, 500.f, &sphere, 3);
    h_clean_memory_for_spheres(&sphere);
    d_allocate_memory_for_spheres(&d_spheres, 3);
    d_clean_memory_for_spheres(&d_spheres);

    h_allocate_memory_for_light_sources(&lights, NUMBER_OF_LIGHTS);
    create_random_light_sources(&lights, NUMBER_OF_LIGHTS);
    h_clean_memory_for_light_sources(&lights);
    d_allocate_memory_for_light_sources(&d_lights, NUMBER_OF_LIGHTS);
    d_clean_memory_for_light_sources(&d_lights);

    GLubyte* h_bitmap = (GLubyte*)malloc(WIDTH * HEIGHT * 3  * sizeof(GLubyte));
    GLubyte* d_bitmap;
    checkCudaErrors(cudaMalloc((void**)&d_bitmap, WIDTH * HEIGHT * 3 * sizeof(GLubyte)));


    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA SPHERES", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}