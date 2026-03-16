find_package(Eigen3 CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)

add_library(core INTERFACE)
target_include_directories(core 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include

    INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(core INTERFACE 
    Eigen3::Eigen 
    nanoflann::nanoflann
    Ceres::ceres
)
