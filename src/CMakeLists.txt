# CMakeList.txt : CMake project for cmake-autogen, include source and define
# project specific logic here.
#

add_executable (CUDA_spheres main.cu
../objects/spheres.cu
../includes/cuda_helper.cu
../objects/light_sources.cu)

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
    ../objects)
