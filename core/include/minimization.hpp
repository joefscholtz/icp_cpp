#pragma once
#include "correspondences.hpp"
#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <vector>

struct ICPResult {
  Eigen::Matrix4d T; // Homogeneous Transformation
  double chi;        // Residual Error
};

using MinimizationFunctionType = std::function<ICPResult(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                                         const std::vector<correspondence_t> &correspondences)>;
auto minimize_point_to_point_svd(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                 const std::vector<correspondence_t> &correspondences) -> ICPResult {
  size_t N = correspondences.size();
  Eigen::Matrix3Xd Ps(3, N);
  Eigen::Matrix3Xd Qs(3, N);

  for (size_t i = 0; i < N; i++) {
    Ps.col(i) = P[correspondences[i].first];
    Qs.col(i) = Q[correspondences[i].second];
  }

  Eigen::Vector3d centroid_P = Ps.rowwise().mean();
  Eigen::Vector3d centroid_Q = Qs.rowwise().mean();

  Eigen::Matrix3Xd P_centered = Ps.colwise() - centroid_P;
  Eigen::Matrix3Xd Q_centered = Qs.colwise() - centroid_Q;

  Eigen::Matrix3d H = P_centered * Q_centered.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  Eigen::Matrix3d R = V * U.transpose();

  // Handle reflection
  if (R.determinant() < 0) {
    V.col(2) *= -1;
    R = V * U.transpose();
  }

  Eigen::Vector3d t = centroid_Q - R * centroid_P;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;

  double chi = (R * Ps + t - Qs).norm();

  return {.T = T, .chi = chi};
}
