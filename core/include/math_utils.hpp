#pragma once
#include <Eigen/Dense>
#include <cstddef>

inline auto transform_vector_points(const std::vector<Eigen::Vector3d> &P, Eigen::Matrix3d R, Eigen::Vector3d t) -> std::vector<Eigen::Vector3d> {
  size_t N = P.size();
  std::vector<Eigen::Vector3d> P_transformed(N);
  for (size_t i = 0; i < N; i++) {
    P_transformed[i] = R * P[i] + t;
  }
  return P_transformed;
}
