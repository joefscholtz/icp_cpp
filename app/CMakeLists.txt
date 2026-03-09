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
    core
    io
)
