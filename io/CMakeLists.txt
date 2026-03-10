find_package(Eigen3 CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

add_library(io STATIC
    src/io.cpp
)
target_include_directories(io 
    # PUBLIC: Consumers of 'io' need access to the public headers.
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include

    # PUBLIC: Consumers also need access to the generated headers in the build directory.
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(io PUBLIC 
    Eigen3::Eigen 
    tinyobjloader::tinyobjloader
)
