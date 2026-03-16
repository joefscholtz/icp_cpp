#pragma once
#include "correspondences.hpp"
#include "time_utils.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <nanoflann.hpp>
#include <optional>
#include <vector>

struct ICPResult {
  Eigen::Matrix4d T{Eigen::Matrix4d::Identity()}; // Homogeneous Transformation
  double chi{0};                                  // Residual Error
};

using MinimizationFunctionType =
    std::function<ICPResult(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                            const std::vector<correspondence_t> &correspondences, std::shared_ptr<std::chrono::duration<double>> duration_ptr)>;

inline auto minimize_point_to_point_svd(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                        const std::vector<correspondence_t> &correspondences,
                                        std::shared_ptr<std::chrono::duration<double>> duration_ptr) -> ICPResult {
  Timer timer;
  // std::cout << "Running minimize_point_to_point_svd" << std::endl;
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

  // TODO: add Eigen::BDCSVD option
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

  double chi = ((R * Ps).colwise() + t - Qs).squaredNorm() / static_cast<double>(N);

  if (duration_ptr != nullptr) {
    *duration_ptr = timer.get_duration();
  }

  return {.T = T, .chi = chi};
}

struct PointToPointError {
  PointToPointError(Eigen::Vector3d p, Eigen::Vector3d q) : p_(p), q_(q) {}

  template <typename T> bool operator()(const T *const angle_axis, const T *const translation, T *residual) const {
    T p[3];
    p[0] = T(p_.x());
    p[1] = T(p_.y());
    p[2] = T(p_.z());

    // Rotate point p using angle-axis representation
    T p_rotated[3];
    ceres::AngleAxisRotatePoint(angle_axis, p, p_rotated);

    // Apply translation and compute residual (difference from q)
    residual[0] = p_rotated[0] + translation[0] - T(q_.x());
    residual[1] = p_rotated[1] + translation[1] - T(q_.y());
    residual[2] = p_rotated[2] + translation[2] - T(q_.z());

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &p, const Eigen::Vector3d &q) {
    return new ceres::AutoDiffCostFunction<PointToPointError, 3, 3, 3>(new PointToPointError(p, q));
  }

  const Eigen::Vector3d p_;
  const Eigen::Vector3d q_;
};

inline auto minimize_point_to_point_ls(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                       const std::vector<correspondence_t> &correspondences,
                                       std::shared_ptr<std::chrono::duration<double>> duration_ptr) -> ICPResult {
  Timer timer;

  // Parameters to optimize: 3 for rotation (Angle-Axis), 3 for translation
  double angle_axis[3] = {0.0, 0.0, 0.0};
  double translation[3] = {0.0, 0.0, 0.0};

  ceres::Problem problem;
  for (const auto &corr : correspondences) {
    ceres::CostFunction *cost_function = PointToPointError::Create(P[corr.first], Q[corr.second]);
    problem.AddResidualBlock(cost_function, nullptr, angle_axis, translation);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-6;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Convert Angle-Axis back to Rotation Matrix
  Eigen::Vector3d aa(angle_axis[0], angle_axis[1], angle_axis[2]);
  double angle = aa.norm();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  if (angle > std::numeric_limits<double>::epsilon()) {
    R = Eigen::AngleAxisd(angle, aa.normalized()).toRotationMatrix();
  }

  Eigen::Vector3d t(translation[0], translation[1], translation[2]);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;

  if (duration_ptr != nullptr) {
    *duration_ptr = timer.get_duration();
  }

  return {.T = T, .chi = summary.final_cost / correspondences.size()};
}
