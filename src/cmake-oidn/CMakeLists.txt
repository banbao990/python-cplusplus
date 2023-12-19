cmake_minimum_required (VERSION 3.20)

set(proj_name "oidn_example")

project (${proj_name} LANGUAGES CXX)

add_subdirectory(pybind11)
pybind11_add_module(${proj_name}
    bind.cpp
    oidn_denoiser.cpp
    oidn_denoiser.h
)

set_property(TARGET ${proj_name} PROPERTY CXX_STANDARD 17)
target_include_directories(${proj_name} PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_directories(${proj_name} PUBLIC ${TORCH_LIBS_INCLUDE_DIRS})
target_link_libraries(${proj_name} PUBLIC ${TORCH_LIBS})

if(WIN32)
    set(oidn_home "${CMAKE_CURRENT_SOURCE_DIR}/oidn/oidn-2.1.0.x64.windows")
    target_include_directories(${proj_name} PUBLIC "${oidn_home}/include")
    target_link_libraries(${proj_name} PUBLIC "${oidn_home}/lib/OpenImageDenoise.lib" "${oidn_home}/lib/OpenImageDenoise_core.lib")
elseif(UNIX)
    # warning: not tested
    message(WARNING, "warning: not tested")
    set(oidn_home "oidn/oidn-2.1.0.x86_64.linux")
    target_include_directories(${proj_name} PUBLIC ${oidn_home}/include)
    target_link_libraries(${proj_name} PUBLIC ${oidn_home}/lib/libOpenImageDenoise_device_cuda.so.2.1.0)
endif()

target_compile_definitions(${proj_name}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})