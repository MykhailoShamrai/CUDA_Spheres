
#include <GLFW/glfw3.h>
//#include "../objects/spheres.cuh"
#include "../includes/cuda_helper.cuh"
#include "../renderers/kernel_renderer.cuh"
#include "../renderers/cpu_renderer.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>

//#include "cmake-test-cuda.h"

#define NUMBER_OF_SPHERES 10
#define NUMBER_OF_LIGHTS 10
#define WIDTH 1200
#define HEIGHT 900

#define THREAD_NUMBER 16


int main(void)
{
    Spheres sphere;
    Spheres d_spheres;
    LightSources lights;
    LightSources d_lights;

    
    
    //cudaMalloc((void**)&d_spheres, sizeof(Spheres));
    //cudaMalloc((void**)&d_lights, sizeof(LightSources));
    //
    //
    //checkCudaErrors(cudaMemcpy(&d_lights, &lights, sizeof(LightSources), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(&d_spheres, &sphere, sizeof(LightSources), cudaMemcpyHostToDevice));


    
    h_allocate_memory_for_spheres(&sphere, NUMBER_OF_SPHERES);
    //create_test_spheres(&sphere);
    create_random_spheres(&sphere, NUMBER_OF_SPHERES);

    h_allocate_memory_for_light_sources(&lights, NUMBER_OF_LIGHTS);
    create_random_light_sources(&lights, NUMBER_OF_LIGHTS);
    

    d_allocate_memory_for_spheres(&d_spheres, NUMBER_OF_SPHERES);
    d_allocate_memory_for_light_sources(&d_lights, NUMBER_OF_LIGHTS);

    checkCudaErrors(cudaMemcpy(d_spheres.x, sphere.x, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.y, sphere.y, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.z, sphere.z, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.ka, sphere.ka, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.ks, sphere.ks, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.kd, sphere.kd, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.R, sphere.R, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.G, sphere.G, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.B, sphere.B, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.alpha, sphere.alpha, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_lights.x, lights.x, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.y, lights.y, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.z, lights.z, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.R, lights.R, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.G, lights.G, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.B, lights.B, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));



    GLubyte* h_bitmap = (GLubyte*)malloc(WIDTH * HEIGHT * 3  * sizeof(GLubyte));
    GLubyte* d_bitmap;
    checkCudaErrors(cudaMalloc((void**)&d_bitmap, WIDTH * HEIGHT * 3 * sizeof(GLubyte)));

    int dim_blocks_x = (WIDTH + THREAD_NUMBER - 1) / THREAD_NUMBER;
    int dim_blocks_y = (HEIGHT + THREAD_NUMBER - 1) / THREAD_NUMBER;

    dim3 blocks(dim_blocks_x, dim_blocks_y);
    dim3 threads(THREAD_NUMBER, THREAD_NUMBER);

    refresh_bitmap << <blocks, threads >> > (d_bitmap, d_spheres, NUMBER_OF_SPHERES, d_lights, NUMBER_OF_LIGHTS, WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    //checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpy(h_bitmap, d_bitmap, WIDTH * HEIGHT * 3 * sizeof(GLubyte), cudaMemcpyDeviceToHost));


    //refresh_bitmap_cpu(h_bitmap, sphere, NUMBER_OF_SPHERES, lights, NUMBER_OF_LIGHTS, WIDTH, HEIGHT);


    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Test Window", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    glViewport(0, 0, WIDTH, HEIGHT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIDTH, 0, HEIGHT, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();




    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, h_bitmap);
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
    glfwTerminate();
    
    

    free(h_bitmap);
    checkCudaErrors(cudaFree(d_bitmap));
    d_clean_memory_for_spheres(&d_spheres);
    h_clean_memory_for_light_sources(&lights);
    d_clean_memory_for_light_sources(&d_lights);
    h_clean_memory_for_spheres(&sphere);
    //checkCudaErrors(cudaFree(&d_spheres));
    //checkCudaErrors(cudaFree(&d_lights));
    return 0;
}