cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(RenderingWithVulkan VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_subdirectory("${CMAKE_SOURCE_DIR}/deps")

add_executable(${PROJECT_NAME} 
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
	"${CMAKE_SOURCE_DIR}/src/Application.hpp"
	"${CMAKE_SOURCE_DIR}/src/Application.cpp"
)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC deps::deps
)

if(WIN32)
    if (TARGET SDL3::SDL3-shared)
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy
                $<TARGET_FILE:SDL3::SDL3-shared> 
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            VERBATIM
        )
    else()
        message(WARNING "SDL3::SDL3-shared target not found. Skipping copy operation...")
    endif()
endif()