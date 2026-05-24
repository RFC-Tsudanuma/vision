# add_definitions(-DAPI_EXPORTS)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# RTX 3090=sm_86, RTX 4090=sm_89, Jetson Orin Nano=sm_87. Generate SASS for
# all three plus forward-compat PTX so the same .so runs on all dev/Jetson
# boxes without JIT "unsupported toolchain" errors.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 87 89)
endif()
enable_language(CUDA)

set(SRCS block.cpp
         calibrator.cpp
         impl.cpp
         rfdetr_impl.cpp
         model.cpp
         postprocess.cpp
         postprocess.cu
         preprocess.cu
         gpu_preprocess.cu
         yololayer.cu
         )

add_library(yolov8_trt SHARED ${SRCS})
target_link_libraries(yolov8_trt PRIVATE nvinfer
                                         cudart
                                 PUBLIC ${OpenCV_LIBS})
target_compile_options(yolov8_trt PRIVATE -w -Wno-dev)
install(TARGETS yolov8_trt DESTINATION lib)
                              
add_executable(yolov8_det yolov8_det.cpp)
target_link_libraries(yolov8_det yolov8_trt nvinfer cudart)
target_compile_options(yolov8_det PRIVATE -w)

add_executable(yolov8_seg yolov8_seg.cpp)
target_link_libraries(yolov8_seg yolov8_trt nvinfer cudart)
target_compile_options(yolov8_seg PRIVATE -w)

install(TARGETS yolov8_det
                yolov8_seg
        DESTINATION lib/${PROJECT_NAME})