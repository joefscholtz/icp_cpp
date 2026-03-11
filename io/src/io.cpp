#include "io.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

auto loadMesh(const std::string &path) -> std::vector<Eigen::Vector3d> {
  std::vector<Eigen::Vector3d> points;
  std::string ext = std::filesystem::path(path).extension().string();

  if (ext == ".ply") {
    try {
      happly::PLYData plyIn(path);
      // happly returns a vector of std::array<double, 3>
      std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
      points.reserve(vPos.size());
      for (const auto &p : vPos) {
        points.emplace_back(p[0], p[1], p[2]);
      }
    } catch (const std::exception &e) {
      std::cerr << "happly error: " << e.what() << std::endl;
    }
  } else if (ext == ".obj") {
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(path, reader_config)) {
      if (!reader.Error().empty()) {
        std::cerr << "TinyObjReader error: " << reader.Error();
      }
      return {};
    }

    auto &attrib = reader.GetAttrib();
    // vertices are stored as a flat float array [x, y, z, x, y, z...]
    points.reserve(attrib.vertices.size() / 3);
    for (size_t v = 0; v < attrib.vertices.size(); v += 3) {
      points.emplace_back(attrib.vertices[v], attrib.vertices[v + 1], attrib.vertices[v + 2]);
    }
  } else {
    std::cerr << "Unsupported file extension: " << ext << std::endl;
  }

  std::cout << "Successfully loaded " << points.size() << " points from " << path << std::endl;
  return points;
}
