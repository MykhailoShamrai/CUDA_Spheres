
#include <GLFW/glfw3.h>
#include "../includes/cuda_helper.cuh"
#include "../renderers/kernel_renderer.cuh"
#include "../renderers/cpu_renderer.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "camera_helpers.cuh"
#include <stdio.h>
#include <cuda_gl_interop.h>
#include <algorithm>
#include <string>
#include <stdexcept>

#define NUMBER_OF_SPHERES 1000
#define NUMBER_OF_LIGHTS 10
//#define WIDTH 1600
//#define HEIGHT 800
#define THREAD_NUMBER 16
#define THREAD_ROTATE_NUMBER 128

#define SENSETIVITY_OF_MOUSE 0.05f

static bool IS_ANIMATED = true;

static int old_width = 1600;
static int old_height = 800;
static int n_width = old_width;
static int n_height = old_height;

static float lastX = old_width / 2.0;
static float lastY = old_height / 2.0;
static bool dragging = false;
static bool light_rotation = false;

static float angle_y_spheres = 0.0f;
static float angle_x_spheres = 0.0f;
static float angle_y_lights = 0.0f;
static float angle_x_lights = 0.0f;

static bool gpu_render = true;

char output_text_buffer[256];

static void animation_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        IS_ANIMATED = !IS_ANIMATED;
    }
    else if (key == GLFW_KEY_LEFT_SHIFT && action == GLFW_PRESS)
    {
        light_rotation = !light_rotation;
    }
    else if (key == GLFW_KEY_C && action == GLFW_PRESS)
    {
        gpu_render = !gpu_render;
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

        if (light_rotation)
        {
            angle_x_lights += xoffset;
            angle_y_lights += yoffset;
        }
        else
        {
            angle_x_spheres += xoffset;
            angle_y_spheres += yoffset;
        }
    }
}

static void framebuffer_size_callback(GLFWwindow*, int new_width, int new_height)
{
    n_width = new_width;
    n_height = new_height;
    glViewport(0, 0, new_width, new_height);
}

char* get_cmd_option(char** begin, char** end, const std::string& option)
{
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char** argv)
{
    int num_spheres = NUMBER_OF_SPHERES;
    int num_lights = NUMBER_OF_LIGHTS;
    
    // Custom number of spheres
    if (cmd_option_exists(argv, argv + argc, "-s"))
    {
        try 
        {
            char* val = get_cmd_option(argv, argv + argc, "-s");
            int num_spheres_tmp = std::stoi(val, 0, 10);
            num_spheres = num_spheres_tmp <= 0 ? NUMBER_OF_SPHERES : num_spheres_tmp;
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "Wrong argument for number of spheres! Default number = %d is set\n", NUMBER_OF_SPHERES);
        }
    }
    
    if (cmd_option_exists(argv, argv + argc, "-l"))
    {
        try
        {
            char* val = get_cmd_option(argv, argv + argc, "-l"); 
            int num_lights_tmp = std::stoi(val, 0, 10);
            num_lights = num_lights_tmp <= 0 ? NUMBER_OF_SPHERES : num_lights_tmp;
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "Wrong argument for number of lights! Default number = %d is set\n", NUMBER_OF_LIGHTS);
        }
    }

    Spheres spheres;
    Spheres d_spheres;
    LightSources lights;
    LightSources d_lights;

    h_allocate_memory_for_spheres(&spheres, num_spheres);
    create_random_spheres(&spheres, num_spheres);

    h_allocate_memory_for_light_sources(&lights, num_lights);
    create_random_light_sources(&lights, num_lights);
   
    d_allocate_memory_for_spheres(&d_spheres, num_spheres);
    d_allocate_memory_for_light_sources(&d_lights, num_lights);

    checkCudaErrors(cudaMemcpy(d_spheres.x_unrotated, spheres.x_unrotated, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.y_unrotated, spheres.y_unrotated, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.z_unrotated, spheres.z_unrotated, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.x, spheres.x, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.y, spheres.y, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.z, spheres.z, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.ka, spheres.ka, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.ks, spheres.ks, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.kd, spheres.kd, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.R, spheres.R, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.G, spheres.G, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.B, spheres.B, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.alpha, spheres.alpha, num_spheres * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spheres.radius, spheres.radius, num_spheres * sizeof(float), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMemcpy(d_lights.x_unrotated, lights.x_unrotated, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.y_unrotated, lights.y_unrotated, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.z_unrotated, lights.z_unrotated, num_lights * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_lights.x, lights.x, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.y, lights.y, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.z, lights.z, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.R, lights.R, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.G, lights.G, num_lights * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lights.B, lights.B, num_lights * sizeof(float), cudaMemcpyHostToDevice));

    float3 camera_pos = make_float3(0, 0, - n_width / 2);
    float* h_bitmap = (float*)malloc(n_width * n_height * 3 * sizeof(float));
    float* d_bitmap;
    checkCudaErrors(cudaMalloc((void**)&d_bitmap, n_width * n_height * 3 * sizeof(float)));


    // Create a window and a context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    if (!glfwInit())
        return -1;

    GLFWwindow* window = glfwCreateWindow(n_width, n_height, "CUDA SPHERES", NULL, NULL);
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

    glGetString(GL_VERSION);

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
    cudaEvent_t start_rotate, stop_rotate;
    cudaEventCreate(&start_rotate);
    cudaEventCreate(&stop_rotate);

    int dim_blocks_x = (n_width + THREAD_NUMBER - 1) / THREAD_NUMBER;
    int dim_blocks_y = (n_height + THREAD_NUMBER - 1) / THREAD_NUMBER;


    // block for framecounting
    int number_of_frames = 0;
    double last_time = glfwGetTime();
    double last_time_anim = glfwGetTime();
    int frame_rate = 0;

    dim3 blocks(dim_blocks_x, dim_blocks_y);
    dim3 threads(THREAD_NUMBER, THREAD_NUMBER);

    int blocks_for_rotation = (num_spheres + THREAD_ROTATE_NUMBER - 1) / THREAD_ROTATE_NUMBER;

    //float3 new_camera_pos = camera_pos;
    while (!glfwWindowShouldClose(window))
    {
       
        number_of_frames++;
        double current_time = glfwGetTime();
        if (IS_ANIMATED)
        {
            double time_diff = current_time - last_time_anim;
            angle_x_spheres += time_diff * 3.0f;
            angle_y_spheres += time_diff * 3.0f;
        }
        last_time_anim = current_time;

        if (current_time - last_time >= 1.0)
        {
            frame_rate = number_of_frames;
            number_of_frames = 0;
            last_time = current_time;
        }
        
        // Check if window was resized
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
        }

        angle_x_spheres = angle_x_spheres > 360.0f ? 0 : angle_x_spheres < -360.0f ? 0 : angle_x_spheres;
        angle_y_spheres = angle_y_spheres > 360.0f ? 0 : angle_y_spheres < -360.0f ? 0 : angle_y_spheres;
        angle_x_lights = angle_x_lights > 360.0f ? 0 : angle_x_lights < -360.0f ? 0 : angle_x_lights;
        angle_y_lights = angle_y_lights > 360.0f ? 0 : angle_y_lights < -360.0f ? 0 : angle_y_lights;


        float elapsed_time = 0;
        float elapsed_time_mem = 0;
        float elapsed_time_rotation = 0;
        // KERNEL PART
        if (gpu_render)
        {
            cudaEventRecord(start_rotate);
            rotate_objects << <blocks_for_rotation, THREAD_ROTATE_NUMBER >> > (d_spheres, d_lights, angle_x_spheres, angle_y_spheres, angle_x_lights, angle_y_lights,
                num_spheres, num_lights);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            cudaEventRecord(stop_rotate);
            cudaEventRecord(start);
            
            unsigned shmem_size = sizeof(unsigned char) * num_spheres;
            refresh_bitmap << <blocks, threads, shmem_size >> > (d_bitmap, d_spheres, num_spheres, d_lights, num_lights,
                n_width, n_height, camera_pos);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            cudaEventRecord(stop);
            
            cudaEventRecord(start_mem);
            checkCudaErrors(cudaMemcpy(h_bitmap, d_bitmap, n_width * n_height * 3 * sizeof(float), cudaMemcpyDeviceToHost));
            cudaEventRecord(stop_mem);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            cudaEventElapsedTime(&elapsed_time_mem, start_mem, stop_mem);
            cudaEventElapsedTime(&elapsed_time_rotation, start_rotate, stop_rotate);
           
        }
        else
        {
            double start_rot = glfwGetTime();
            for (int i = 0; i < num_spheres; i++)
            {
                rotate_positions(&spheres.x[i], &spheres.z[i], &spheres.x_unrotated[i], &spheres.z_unrotated[i], angle_x_spheres);
                rotate_positions(&spheres.y[i], &spheres.z[i], &spheres.y_unrotated[i], &spheres.z[i], angle_y_spheres);
            }
            for (int i = 0; i < num_lights; i++)
            {
                rotate_positions(&lights.x[i], &lights.z[i], &lights.x_unrotated[i], &lights.z_unrotated[i], angle_x_lights);
                rotate_positions(&lights.y[i], &lights.z[i], &lights.y_unrotated[i], &lights.z[i], angle_y_lights);
            }
            double end_rot = glfwGetTime();
            elapsed_time_rotation = (end_rot - start_rot) * 1000;
            double start_render = glfwGetTime();
            refresh_bitmap_cpu(h_bitmap, spheres, num_spheres, lights, num_lights, n_width,
                n_height, camera_pos);
            double end_render = glfwGetTime();
            elapsed_time = (end_render - start_render) * 1000;
        }
        
        sprintf(output_text_buffer, "FPS: %d :: MEMORY COPY %.4f :: FRAME GENERATION FUNCTION :: %.4f :: ROTATION %.4f", frame_rate, elapsed_time_mem, elapsed_time, elapsed_time_rotation);
        
        glfwSetWindowTitle(window, output_text_buffer);
        glClear(GL_COLOR_BUFFER_BIT);

        
        glDrawPixels(n_width, n_height, GL_RGB, GL_FLOAT, h_bitmap);
        glfwSwapBuffers(window);

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