# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
  if(EXISTS /usr/include/aarch64-linux-gnu/NvInferVersion.h)
    include_directories(/usr/include/aarch64-linux-gnu)
    link_directories(/usr/lib/aarch64-linux-gnu)
  endif()
else()
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)
  if(EXISTS /usr/include/x86_64-linux-gnu/NvInferVersion.h)
    include_directories(/usr/include/x86_64-linux-gnu)
    link_directories(/usr/lib/x86_64-linux-gnu)
  else()
    include_directories(/usr/local/TensorRT-8.6.1.6/include)
    link_directories(/usr/local/TensorRT-8.6.1.6/lib)
  endif()
endif()

add_subdirectory(trt)
if (ENABLE_ONNX)
  add_subdirectory(onnx)
  add_definitions(-DENABLE_ONNX)
endif()

add_library(yolov8_model SHARED detector.cc
                                segmentor.cc)

target_link_libraries(yolov8_model PUBLIC yolov8_trt
                                            $<$<BOOL:${ENABLE_ONNX}>:yolov8_onnx>
                                          ${YAML_CPP_LIBRARIES})
target_compile_options(yolov8_model PRIVATE -w)

install(TARGETS yolov8_model DESTINATION lib)