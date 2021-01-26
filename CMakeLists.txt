# Set  the minimum  required version  of cmake  for a  project.
cmake_minimum_required(VERSION 2.6)

# C++11
SET(CMAKE_CXX_FLAGS "-std=c++0x")

# Azure Kinect / OpenCV
FIND_PACKAGE(k4a REQUIRED)
#FIND_PACKAGE(k4abt REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
set(VTK_DIR "/home/hurel/codes/openCV/VTK/build")
find_package(VTK REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#include directory
include_directories(${PROJECT_SOURCE_DIR}/include)
include(${VTK_USE_FILE})

# locate source files
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

# Add an executable to the project using the specified source files.

add_executable(IR_tracker main.cpp ${headers} ${sources})
# link_directories(/home/hurel/Azure-Kinect-Sensor-SDK/build/lib)
target_link_libraries(IR_tracker ${OpenCV_LIBS} ${VTK_LIBRARIES} k4a Eigen3::Eigen)
