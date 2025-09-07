cmake_minimum_required(VERSION 3.20)
project(neuralT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg toolchain assumed via -DCMAKE_TOOLCHAIN_FILE, but this file is agnostic

# --- Dependencies (vcpkg CONFIG) ---
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# --- Sources ---
# Keep explicit to avoid accidentally picking CMake's compiler-id sources
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(neuralT ${SOURCES})
if (MSVC)
  target_compile_options(neuralT PRIVATE /utf-8)
endif()

# --- Includes ---
target_include_directories(neuralT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- ImGui backends (headers live under include/imgui/backends in vcpkg) ---
# Prefer vcpkg-installed headers; add include so #include <imgui/backends/...> works.
if(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
    set(_IMGUI_INC_DIR "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}/include/imgui")
    if(EXISTS "${_IMGUI_INC_DIR}/backends/imgui_impl_glfw.h")
        target_include_directories(neuralT PRIVATE "${_IMGUI_INC_DIR}")
    endif()
endif()

# Loader define for OpenGL3 backend when using GLEW
# Enable Dear ImGui integration in code and use GLEW loader
target_compile_definitions(neuralT PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLEW USE_IMGUI)

# Try to add backend implementations if we can locate them in vcpkg headers (some ports ship .cpp alongside headers).
set(_IMGUI_BACKENDS_CPP_GLFW "")
set(_IMGUI_BACKENDS_CPP_OGL3 "")

if(DEFINED _IMGUI_INC_DIR)
    if(EXISTS "${_IMGUI_INC_DIR}/backends/imgui_impl_glfw.cpp")
        set(_IMGUI_BACKENDS_CPP_GLFW "${_IMGUI_INC_DIR}/backends/imgui_impl_glfw.cpp")
    endif()
    if(EXISTS "${_IMGUI_INC_DIR}/backends/imgui_impl_opengl3.cpp")
        set(_IMGUI_BACKENDS_CPP_OGL3 "${_IMGUI_INC_DIR}/backends/imgui_impl_opengl3.cpp")
    endif()
endif()

# Fallback: project-local vendor path (copy the two .cpp there if needed)
if(NOT _IMGUI_BACKENDS_CPP_GLFW OR NOT _IMGUI_BACKENDS_CPP_OGL3)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui_backends/imgui_impl_glfw.cpp" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui_backends/imgui_impl_opengl3.cpp")
        set(_IMGUI_BACKENDS_CPP_GLFW "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui_backends/imgui_impl_glfw.cpp")
        set(_IMGUI_BACKENDS_CPP_OGL3 "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui_backends/imgui_impl_opengl3.cpp")
        # also include dir for local headers if you keep backends/ next to them
        target_include_directories(neuralT PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui_backends")
    endif()
endif()

if(_IMGUI_BACKENDS_CPP_GLFW AND _IMGUI_BACKENDS_CPP_OGL3)
    target_sources(neuralT PRIVATE
        "${_IMGUI_BACKENDS_CPP_GLFW}"
        "${_IMGUI_BACKENDS_CPP_OGL3}"
    )
endif()

# --- Linkage ---
target_link_libraries(neuralT PRIVATE
    glfw
    GLEW::GLEW
    glm::glm
    imgui::imgui
    opengl32
)
