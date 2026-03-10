include(FetchContent)
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v1.3.0
)
FetchContent_MakeAvailable(polyscope)

find_package(nfd CONFIG REQUIRED)

add_executable(icp_app src/main.cpp)

target_link_libraries(icp_app PUBLIC 
    polyscope 
    nfd::nfd
    core
    io
)
