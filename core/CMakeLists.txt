find_package(Eigen3 CONFIG REQUIRED)

add_library(core INTERFACE
    # src/icp.cpp
)
target_include_directories(core 
    # PUBLIC: Consumers of 'core' need access to the public headers.
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include

    # PUBLIC: Consumers also need access to the generated headers in the build directory.
    INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(core INTERFACE 
    Eigen3::Eigen 
)
