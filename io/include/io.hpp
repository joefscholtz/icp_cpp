#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "happly.h"
#include "tiny_obj_loader.h"

auto loadMesh(const std::string &path) -> std::vector<Eigen::Vector3d>;
