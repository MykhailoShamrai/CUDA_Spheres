# CMakeList.txt : CMake project for cmake-autogen, include source and define
# project specific logic here.
#

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

file(GLOB_RECURSE SOURCES
        "${CMAKE_SOURCE_DIR}/includes/*.cu"
        "${CMAKE_SOURCE_DIR}/objects/*.cu"
        "${CMAKE_SOURCE_DIR}/renderers/*.cu"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_executable(CUDA_spheres
    ${SOURCES}
)

set_target_properties(CUDA_spheres PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

find_package(OpenGL REQUIRED)

target_link_libraries(CUDA_spheres PRIVATE
        glfw
        OpenGL::GL
)

target_include_directories(CUDA_spheres PRIVATE
        ${CMAKE_SOURCE_DIR}/includes
        ${CMAKE_SOURCE_DIR}/objects
        ${CMAKE_SOURCE_DIR}/renderers
        ${CMAKE_SOURCE_DIR}/src
)