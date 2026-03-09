cmake_minimum_required(VERSION 3.14)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(icp_cpp)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

find_package(Eigen3 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v1.3.0
)
FetchContent_MakeAvailable(polyscope)

add_executable(icp_app src/main.cpp)

target_link_libraries(icp_app PUBLIC 
    polyscope 
    Eigen3::Eigen 
)
