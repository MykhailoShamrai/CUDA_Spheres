# CMakeList.txt : CMake project for cmake-autogen, include source and define
# project specific logic here.
#

add_library(renderers_helper STATIC 
    ../renderers/renderers_helper.cu
    ../renderers/renderers_helper.cuh
)
set_target_properties(renderers_helper PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(cpu_renderer STATIC ../renderers/cpu_renderer.cuh ../renderers/cpu_renderer.cu)

set_target_properties(cpu_renderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cpu_renderer PRIVATE renderers_helper)

add_library(kernel_renderer STATIC ../renderers/kernel_renderer.cuh ../renderers/kernel_renderer.cu)

set_target_properties(kernel_renderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(kernel_renderer PRIVATE renderers_helper)

add_executable (CUDA_spheres main.cu
../objects/spheres.cu
../includes/cuda_helper.cu
../objects/light_sources.cu
camera_helpers.cu
)

target_link_libraries(CUDA_spheres PRIVATE cpu_renderer kernel_renderer renderers_helper)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDA_spheres PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

# GLFW Configuration
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Add source to this project's executable.


find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenGL REQUIRED)
# Include directories
target_include_directories(CUDA_spheres PRIVATE ../external/glfw/include)
target_link_libraries(CUDA_spheres PRIVATE glfw OpenGL::GL)
include_directories(
    ../includes
    ../objects
    ../renderers)
