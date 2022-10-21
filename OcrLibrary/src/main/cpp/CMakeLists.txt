cmake_minimum_required(VERSION 3.4.1)
project(RapidOcr)
option(OCR_VULKAN "RapidOcr Enable Vulkan" OFF)
#set(OCR_VULKAN ON)
if (OCR_VULKAN)
    set(ncnn_DIR "${CMAKE_SOURCE_DIR}/../ncnn-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
else ()
    set(ncnn_DIR "${CMAKE_SOURCE_DIR}/../ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
endif ()
find_package(ncnn REQUIRED)

## opencv 库
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../../sdk/native/jni")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

include_directories(include)
file(GLOB OCR_SRC src/*.cpp)
set(OCR_COMPILE_CODE ${OCR_SRC})

add_library(RapidOcr SHARED ${OCR_COMPILE_CODE})

target_link_libraries(
        RapidOcr
        ncnn
        ${OpenCV_LIBS}
        jnigraphics)

if (OCR_VULKAN)
    target_compile_definitions(RapidOcr PRIVATE __VULKAN__)
endif ()