# icp_cpp

A modular, high-performance Iterative Closest Point (ICP) framework implemented in C++20. This project decouples the correspondence search from the error minimization step, allowing for easy benchmarking of various modern ICP variants.

## Features & Algorithms

The implementation is designed to be **header-only** and independent of the GUI, making it easy to integrate the math core into other robotics projects.

### 1. Correspondence Search (Registration)
- **Brute Force:** Naive $O(N^2)$ nearest neighbor search.
- **KD-Tree (nanoflann):** High-performance spatial indexing for fast $O(\log N)$ neighbor lookups.

### 2. Minimization Variants
- **Point-to-Point (SVD):** Classic closed-form solution using Singular Value Decomposition.
- **Point-to-Point (Non-Linear LS):** Robust optimization using the **Ceres Solver**.
- **Point-to-Plane:** Minimizes the distance between points and the surface planes of the target cloud, significantly improving convergence on flat geometries.
- **Generalized ICP (GICP):** A distribution-to-distribution approach that utilizes local surface covariances to model the underlying geometry of both clouds.

### 3. Advanced Optimization
- **Multi-Scale Downsampling:** Mitigates local minima by performing coarse-to-fine registration. It uses voxel-grid downsampling scaled automatically based on the cloud's dimensions.
- **Cumulative Scan-to-Map:** Option to register new scans against a growing global map of previous clouds.
- **Outlier Rejection:** Integrated Huber loss functions via Ceres to handle noisy sensor data and misalignments.

## Dependencies

- **C++20** compatible compiler
- **Cmake** & **Ninja**
- **just** (Optional task runner)
- **VCPKG** (Included as a submodule)

### Linux System Dependencies

```bash
sudo apt install autoconf autoconf-archive automake libtool
```

## Build

Clone the repo and its submodules (Ceres, Eigen, Polyscope, nanoflann, and vcpkg):

```bash
git clone --recurse-submodules git@github.com:joefscholtz/icp_cpp.git
```

### 1. Bootstrap vcpkg

Initialize the package manager to fetch libraries:

- Windows: `3rd_party\vcpkg\bootstrap-vcpkg.bat`
- Linux/MacOS: `3rd_party/vcpkg/bootstrap-vcpkg.sh`

### 2. Compile

```bash
cmake --preset default
cmake --build --preset default --parallel
```

or simply use: `just build`

## Usage

Run the demo application to visualize the Stanford Bunny 🐇 registration:

```bash
./build/app/icp_app
```

or simply use: `just run`

## License

MIT
