find_package(Eigen3 CONFIG REQUIRED)

add_library(core STATIC
    src/main.cpp
)
target_include_directories(core 
    # PUBLIC: Consumers of 'core' need access to the public headers.
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include

    # PUBLIC: Consumers also need access to the generated headers in the build directory.
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)
