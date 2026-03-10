# icp_cpp

## Dependencies

- Cmake
- Ninja
- just (Optional)

### Linux Dependencies

autoconf autoconf-archive automake libtool

## Build

Clone de repo and its submodules

```bash
git clone --recurse-submodules git@github.com:joefscholtz/icp_cpp.git
```

First, bootstrap vcpkg to find the dependencies. In Windows, run `<project_path>\3rd_party\vcpkg\bootstrap-vcpkg.bat`, or in Linux/MacOs `<project_path>/3rd_party/vcpkg/bootstrap-vcpkg.sh`. Then build the project manually

```bash
cmake --preset default
cmake --build --preset default --parallel
```

or with just

```bash
just build
```

## Usage

Run

```bash
./build/app/icp_app
```

or with just

```bash
just run
```
