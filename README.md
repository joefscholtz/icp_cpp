# icp_cpp

## Dependencies

- Cmake
- Ninja
- just (Optional)

## Build

Clone de repo and its submodules

```bash
git clone --recurse-submodules git@github.com:joefscholtz/icp_cpp.git
```

Build the project manually

```bash
cmake --preset default
cmake --build --preset default --parallel
```

Or with just

```bash
just build
```

## Usage

Run

```bash
./build/app/icp_app
```

Or with just

```bash
just run
```
