
#define NUMBER_OF_SPHERES 10
#include <GLFW/glfw3.h>
#include "../objects/sphere.cuh"
//#include "cmake-test-cuda.h"


int main(void)
{
    Spheres sphere;

    h_allocate_memory_for_spheres(&sphere, NUMBER_OF_SPHERES);
    h_clean_memory_for_spheres(&sphere, NUMBER_OF_SPHERES);


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