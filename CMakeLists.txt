cmake_minimum_required(VERSION 3.1)
project(pointmesh)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)

# Add your project files
add_executable(pointmesh-gui gui/main.cpp)
target_link_libraries(pointmesh-gui igl::core igl::opengl_glfw)

add_executable(pointmesh cli/main.cpp)
target_link_libraries(pointmesh igl::core)
