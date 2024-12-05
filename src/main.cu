
#include <GLFW/glfw3.h>
#include "../includes/cuda_helper.cuh"
#include "../renderers/kernel_renderer.cuh"
#include "../renderers/cpu_renderer.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "camera_helpers.cuh"
#include <stdio.h>


#define NUMBER_OF_SPHERES 1000
#define NUMBER_OF_LIGHTS 10
//#define WIDTH 1600
//#define HEIGHT 800
#define THREAD_NUMBER 16

#define SENSETIVITY_OF_MOUSE 0.05f

static bool IS_ANIMATED = true;

static int old_width = 1600;
static int old_height = 800;
static int n_width = old_width;
static int n_height = old_height;


static float lastX = old_width / 2.0;
static float lastY = old_height / 2.0;
static bool dragging = false;

static float angle_y = 0.0f;
static float angle_x = 0.0f;




static void animation_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        IS_ANIMATED = !IS_ANIMATED;
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            lastX = xpos;
            lastY = ypos;
            dragging = true;
            
        }
        else if (action == GLFW_RELEASE)
        {
            dragging = false;
        }
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (dragging)
    {
        // Calculate mouse movement offsets
        float xoffset = lastX - xpos;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;

        xoffset *= SENSETIVITY_OF_MOUSE;
        yoffset *= SENSETIVITY_OF_MOUSE;

        angle_x += xoffset;
        angle_y += yoffset;
    }
}

static void framebuffer_size_callback(GLFWwindow*, int new_width, int new_height)
{
    //old_width = n_width;
    //old_height = n_height;
    n_width = new_width;
    n_height = new_height;
    glViewport(0, 0, new_width, new_height);
}


int main(void)
{
    Spheres spheres;
    Spheres d_spheres;
    LightSources lights;
    LightSources d_lights;

    h_allocate_memory_for_spheres(&spheres, NUMBER_OF_SPHERES);
    create_random_spheres(&spheres, NUMBER_OF_SPHERES);

    h_allocate_memory_for_light_sources(&lights, NUMBER_OF_LIGHTS);
    create_random_light_sources(&lights, NUMBER_OF_LIGHTS);
    

    d_allocate_memory_for_spheres(&d_spheres, NUMBER_OF_SPHERES);
    d_allocate_memory_for_light_sources(&d_lights, NUMBER_OF_LIGHTS);

    float* unrotated_x_spheres = (float*)malloc(sizeof(float) * NUMBER_OF_SPHERES);
    float* unrotated_y_spheres = (float*)malloc(sizeof(float) * NUMBER_OF_SPHERES);
    float* unrotated_z_spheres = (float*)malloc(sizeof(float) * NUMBER_OF_SPHERES);
    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        unrotated_x_spheres[i] = spheres.x[i];
        unrotated_y_spheres[i] = spheres.y[i];
        unrotated_z_spheres[i] = spheres.z[i];
    }

    float* unrotated_x_lights = (float*)malloc(sizeof(float) * NUMBER_OF_LIGHTS);
    float* unrotated_y_lights = (float*)malloc(sizeof(float) * NUMBER_OF_LIGHTS);
    float* unrotated_z_lights = (float*)malloc(sizeof(float) * NUMBER_OF_LIGHTS);
    for (int i = 0; i < NUMBER_OF_LIGHTS; i++)
    {
        unrotated_x_lights[i] = lights.x[i];
        unrotated_y_lights[i] = lights.y[i];
        unrotated_z_lights[i] = lights.z[i];
    }


    checkCudaErrors(cudaMemcpy(d_spheres.x, spheres.x, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.y, spheres.y, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.z, spheres.z, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.ka, spheres.ka, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.ks, spheres.ks, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.kd, spheres.kd, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.R, spheres.R, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.G, spheres.G, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.B, spheres.B, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.alpha, spheres.alpha, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.radius, spheres.radius, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_lights.x, lights.x, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.y, lights.y, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.z, lights.z, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.R, lights.R, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.G, lights.G, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.B, lights.B, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));

    float3 camera_pos = make_float3(0, 0, - n_width / 2);


    float* h_bitmap = (float*)malloc(n_width * n_height * 3 * sizeof(float));
    float* d_bitmap;
    checkCudaErrors(cudaMalloc((void**)&d_bitmap, n_width * n_height * 3 * sizeof(float)));

    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    GLFWwindow* window = glfwCreateWindow(n_width, n_height, "Test Window", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glViewport(0, 0, n_width, n_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, n_width, 0, n_height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


    glfwSetKeyCallback(window, animation_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    


    // Initialisation of timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t start_mem, stop_mem;
    cudaEventCreate(&start_mem);
    cudaEventCreate(&stop_mem);

    // 
    int dim_blocks_x = (n_width + THREAD_NUMBER - 1) / THREAD_NUMBER;
    int dim_blocks_y = (n_height + THREAD_NUMBER - 1) / THREAD_NUMBER;

    dim3 blocks(dim_blocks_x, dim_blocks_y);
    dim3 threads(THREAD_NUMBER, THREAD_NUMBER);
    float3 new_camera_pos = camera_pos;
    while (!glfwWindowShouldClose(window))
    {
        // Important for resizing a window
        if (n_width != old_width || n_height != old_height)
        {
            free(h_bitmap);
            checkCudaErrors(cudaFree(d_bitmap));
            h_bitmap = (float*)malloc(n_width * n_height * 3 * sizeof(float));
            checkCudaErrors(cudaMalloc((void**)&d_bitmap, n_width * n_height * 3 * sizeof(float)));
            old_height = n_height;
            old_width = n_width;
            camera_pos = make_float3(0, 0, -n_width / 2);
            dim_blocks_x = (n_width + THREAD_NUMBER - 1) / THREAD_NUMBER;
            dim_blocks_y = (n_height + THREAD_NUMBER - 1) / THREAD_NUMBER;
            blocks = dim3(dim_blocks_x, dim_blocks_y);
            printf("%d, %d\n", n_width, n_height);
        }


        angle_x = angle_x > 360.0f ? 0 : angle_x < -360.0f ? 0 : angle_x;
        angle_y = angle_y > 360.0f ? 0 : angle_y < -360.0f ? 0 : angle_y;
        
        if (IS_ANIMATED)
        {
            angle_x += 0.2f;
            angle_y += 0.2f;
        }

        cudaEventRecord(start);
        rotate_positions(spheres.x, spheres.z, unrotated_x_spheres, unrotated_z_spheres, angle_x, NUMBER_OF_SPHERES);
        rotate_positions(spheres.y, spheres.z, unrotated_y_spheres, spheres.z, angle_y, NUMBER_OF_SPHERES);
        cudaEventRecord(start_mem);
        checkCudaErrors(cudaMemcpy(d_spheres.x, spheres.x, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_spheres.y, spheres.y, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_spheres.z, spheres.z, NUMBER_OF_SPHERES * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_lights.x, lights.x, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_lights.y, lights.y, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_lights.z, lights.z, NUMBER_OF_LIGHTS * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(stop_mem);

        unsigned shmem_size = sizeof(unsigned char) * NUMBER_OF_SPHERES;
        refresh_bitmap << <blocks, threads, shmem_size >> > (d_bitmap, d_spheres, NUMBER_OF_SPHERES, d_lights, NUMBER_OF_LIGHTS, n_width, n_height, camera_pos);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaDeviceSynchronize());
        cudaEventRecord(stop);


        checkCudaErrors(cudaMemcpy(h_bitmap, d_bitmap, n_width * n_height * 3 * sizeof(float), cudaMemcpyDeviceToHost));

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("time for generation of frame: %f\n", elapsed_time);
        cudaEventElapsedTime(&elapsed_time, start_mem, stop_mem);
        printf("time for memory copying: %f\n", elapsed_time);

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glDrawPixels(n_width, n_height, GL_RGB, GL_FLOAT, h_bitmap);
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

    }
    glfwTerminate();
    
    

    // cleaning 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_mem);
    cudaEventDestroy(stop_mem);

    free(h_bitmap);
    checkCudaErrors(cudaFree(d_bitmap));
    d_clean_memory_for_spheres(&d_spheres);
    h_clean_memory_for_light_sources(&lights);
    d_clean_memory_for_light_sources(&d_lights);
    h_clean_memory_for_spheres(&spheres);
    return 0;
}