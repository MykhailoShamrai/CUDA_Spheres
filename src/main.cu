
#define NUMBER_OF_SPHERES 10
#define NUMBER_OF_LIGHTS 2
#include <GLFW/glfw3.h>
#include "../objects/spheres.cuh"
#include "../objects/light_sources.cuh"
//#include "cmake-test-cuda.h"


int main(void)
{
    Spheres sphere;
    Spheres d_spheres;
    LightSources lights;
    LightSources d_lights;
    h_allocate_memory_for_spheres(&sphere, NUMBER_OF_SPHERES);
    create_random_spheres(&sphere, NUMBER_OF_SPHERES);
    h_clean_memory_for_spheres(&sphere);
    d_allocate_memory_for_spheres(&d_spheres, NUMBER_OF_SPHERES);
    d_clean_memory_for_spheres(&d_spheres);

    h_allocate_memory_for_light_sources(&lights, NUMBER_OF_LIGHTS);
    create_random_light_sources(&lights, NUMBER_OF_LIGHTS);
    h_clean_memory_for_light_sources(&lights);
    d_allocate_memory_for_light_sources(&d_lights, NUMBER_OF_LIGHTS);
    d_clean_memory_for_light_sources(&d_lights);


    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
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