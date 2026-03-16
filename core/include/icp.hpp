#pragma once
#include "correspondences.hpp"
#include "math_utils.hpp"
#include "minimization.hpp"
#include "time_utils.hpp"
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

using VisualizationFunctionType = std::function<void(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                                     std::vector<correspondence_t> &correspondences, size_t P_idx, size_t Q_idx)>;

template <typename V>
inline auto icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
                MinimizationFunctionType &minimization_fn, std::optional<V> visualization_fn, std::optional<const Eigen::Matrix4d> T0,
                std::shared_ptr<ICPDuration> icp_duration, const size_t iterations) -> ICPResult {

  // std::cout << "P.size(): " << P.size() << std::endl;
  // std::cout << "Q.size(): " << Q.size() << std::endl;

  Eigen::Matrix4d T = T0.value_or(Eigen::Matrix4d::Identity());
  std::vector<Eigen::Vector3d> normals_P = estimate_normals(P);
  std::vector<Eigen::Vector3d> normals_Q = estimate_normals(Q);
  std::vector<Eigen::Vector3d> P_curr;
  std::vector<correspondence_t> correspondences;
  ICPResult min_result;

  std::shared_ptr<std::chrono::duration<double>> min_duration_ptr{nullptr}, corr_duration_ptr{nullptr};
  bool tracking_time = icp_duration != nullptr;
  if (tracking_time) {
    corr_duration_ptr = std::make_shared<std::chrono::duration<double>>();
    min_duration_ptr = std::make_shared<std::chrono::duration<double>>();
  }

  for (size_t i = 0; i < iterations; i++) {
    P_curr = transform_vector_points(P, T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
    correspondences = correspondence_fn(P_curr, Q, corr_duration_ptr, std::nullopt);
    min_result = minimization_fn(P_curr, Q, normals_P, normals_Q, correspondences, true, min_duration_ptr);

    if (tracking_time) {
      icp_duration->correspondence_duration += *corr_duration_ptr;
      icp_duration->minimization_duration += *min_duration_ptr;
    }

    T = min_result.T * T;

    if (visualization_fn) {
      (*visualization_fn)(P_curr, Q, correspondences);
    }
  }

  if (tracking_time) {
    icp_duration->correspondence_duration /= iterations;
    icp_duration->minimization_duration /= iterations;
    icp_duration->icp_duration = icp_duration->correspondence_duration + icp_duration->minimization_duration;
  }

  return {.T = T, .chi = min_result.chi};
}

inline auto frame_to_frame_icp(std::vector<std::vector<Eigen::Vector3d>> &point_clouds, CorrespondenceFunctionType &correspondence_fn,
                               MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
                               std::optional<const Eigen::Matrix4d> T0, bool start_transform_with_last_estimate,
                               std::shared_ptr<ICPDuration> icp_duration, const size_t iterations, const size_t num_scales = 1,
                               bool cumulative = false) -> ICPResult {

  ICPResult curr_icp;
  Eigen::Matrix4d T_ini = Eigen::Matrix4d::Identity();
  double extent = calculate_cloud_extent(point_clouds[0]);

  for (size_t i = 0; i < point_clouds.size() - 1; i++) {
    size_t P_idx = i + 1, Q_idx = i;

    std::vector<Eigen::Vector3d> target_map;
    if (cumulative) {
      target_map.clear();
      for (size_t j = 0; j <= Q_idx; ++j) {
        target_map.insert(target_map.end(), point_clouds[j].begin(), point_clouds[j].end());
      }
    }
    const std::vector<Eigen::Vector3d> &Q_target = cumulative ? target_map : point_clouds[Q_idx];

    Eigen::Matrix4d T_scale_accum = start_transform_with_last_estimate ? T_ini : Eigen::Matrix4d::Identity();

    for (int s = 0; s < num_scales; ++s) {
      double voxel_size = 0.0;
      if (num_scales > 1) {
        double ratio = 1.0 - (static_cast<double>(s) / (num_scales - 1));
        voxel_size = (s == num_scales - 1) ? 0.0 : (0.1 * extent * ratio);
      }

      auto P_scaled = voxel_downsample(point_clouds[P_idx], voxel_size);
      auto Q_scaled = voxel_downsample(Q_target, voxel_size);

      auto wrapped_viz = [visualization_fn, P_idx, Q_idx](const auto &p, const auto &q, auto &c) {
        if (visualization_fn)
          (*visualization_fn)(p, q, c, P_idx, Q_idx);
      };

      size_t scale_iters = std::max<size_t>(1, iterations / num_scales);

      std::vector<Eigen::Vector3d> nP = estimate_normals(P_scaled);
      std::vector<Eigen::Vector3d> nQ = estimate_normals(Q_scaled);

      curr_icp =
          icp(P_scaled, Q_scaled, correspondence_fn, minimization_fn, std::make_optional(wrapped_viz), T_scale_accum, icp_duration, scale_iters);

      T_scale_accum = curr_icp.T;
    }

    point_clouds[P_idx] = transform_vector_points(point_clouds[P_idx], curr_icp.T.block<3, 3>(0, 0), curr_icp.T.block<3, 1>(0, 3));

    if (start_transform_with_last_estimate)
      T_ini = curr_icp.T;
  }
  return curr_icp;
}
