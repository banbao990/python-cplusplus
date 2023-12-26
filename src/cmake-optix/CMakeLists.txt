cmake_minimum_required (VERSION 3.20)

set(proj_name "cmake_optix_example")

project (${proj_name} LANGUAGES CXX)

add_subdirectory(pybind11)
pybind11_add_module(${proj_name}
    bind.cpp
    denoiser.cpp
    optix_helper.cpp
    optix_helper.h
    denoiser.h
)

set_property(TARGET ${proj_name} PROPERTY CXX_STANDARD 17)
target_include_directories(${proj_name} PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_directories(${proj_name} PUBLIC ${TORCH_LIBS_INCLUDE_DIRS})
target_link_libraries(${proj_name} PUBLIC ${TORCH_LIBS})

target_include_directories(${proj_name} PUBLIC ${OPTIX_INCLUDE_DIR})

target_compile_definitions(${proj_name}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})