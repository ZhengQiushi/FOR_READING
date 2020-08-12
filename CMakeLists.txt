cmake_minimum_required(VERSION 2.8)

project(tjsp_attack_2020)

set(CMAKE_CXX_STANDARD 14)  # for chrono
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -pthread")

set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-std=c++14)

#====================== OpenCV Config ============================
#set(OpenCV_DIR /home/youngxi/opt/OpenCV/opencv-3.4.9/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package (Threads REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#=================================================================

# Add Tensorflow headers location to your include paths
include_directories(./tensorflow/include)
include_directories(./tensorflow/include/src)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
# Find Boost
find_package(Boost REQUIRED COMPONENTS thread)
if (!Boost_FOUND)
    message("-- Found Boost NOT FOUND")
endif ()

#======================== serial ==============================
include_directories(/home/truth/opt/serial/include)
#=================================================================

#======================== 迈德威视 SDK ==============================
include_directories(/home/truth/mindvision/include)
#=================================================================

#======================== Dahua SDK ==============================
include_directories(/opt/DahuaTech/MVviewer/include)
link_libraries(/opt/DahuaTech/MVviewer/lib -lMVSDK)
#=================================================================


include_directories(source include)

add_executable(test main.cpp)

target_link_libraries(test /opt/DahuaTech/MVviewer/lib/libMVSDK.so)
target_link_libraries(test ${OpenCV_LIBS})

target_link_libraries(test /home/truth/opt/serial/build/devel/lib/libserial.so)
target_link_libraries(test ${serial_LIBRARIES})
target_link_libraries(test ${Boost_LIBRARIES} usb-1.0)
target_link_libraries(test /home/truth/ClionProjects/RM/sp/attack2020_tf/tensorflow/libtensorflow_cc.so
                           /home/truth/ClionProjects/RM/sp/attack2020_tf/tensorflow/libtensorflow_framework.so.2)


# FIND_PACKAGE(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     message("OPENMP FOUND")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif ()

