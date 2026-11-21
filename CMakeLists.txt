cmake_minimum_required(VERSION 3.24)
project(adakernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(adakernel SHARED
    native/quant_cpu.cpp
    native/cuda/device.cu
    native/cuda/megakernel.cu
    native/cuda/qwen35_model.cu
)
target_include_directories(adakernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(adakernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda)
target_link_libraries(adakernel PRIVATE CUDA::cublas CUDA::cudart)
target_compile_options(adakernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info -Xcompiler=/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
)
set_target_properties(adakernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    OUTPUT_NAME adakernel
)
