#pragma once
#include "correspondences.hpp"
#include "math_utils.hpp"
#include "minimization.hpp"
#include "time_utils.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

using VisualizationFunctionType =
    std::function<void(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, std::vector<correspondence_t> &correspondences,
                       size_t current_pair_idx, const Eigen::Matrix4d &current_T)>;

template <typename V>
inline auto icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
                MinimizationFunctionType &minimization_fn, std::optional<V> visualization_fn, std::optional<const Eigen::Matrix4d> T0,
                std::shared_ptr<ICPDuration> icp_duration, const size_t iterations) -> ICPResult {

  Eigen::Matrix4d T = T0.value_or(Eigen::Matrix4d::Identity());
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
    min_result = minimization_fn(P_curr, Q, correspondences, min_duration_ptr);

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
                               std::optional<const Eigen::Matrix4d> T0, std::shared_ptr<ICPDuration> icp_duration, const size_t iterations,
                               bool cumulative = false) -> ICPResult {

  ICPResult curr_icp;
  Eigen::Matrix4d global_T = T0.value_or(Eigen::Matrix4d::Identity());

  std::vector<std::vector<Eigen::Vector3d>> current_clouds = point_clouds;
  std::vector<Eigen::Vector3d> target_map;

  for (size_t i = 0; i < point_clouds.size() - 1; i++) {
    auto &P_source = current_clouds[i];

    if (cumulative) {
      // Merge all previous clouds into one target map
      target_map.clear();
      for (size_t j = 0; j <= i; ++j) {
        target_map.insert(target_map.end(), current_clouds[j].begin(), current_clouds[j].end());
      }
    }
    const std::vector<Eigen::Vector3d> &Q_target = cumulative ? target_map : current_clouds[i + 1];

    auto wrapped_viz = [visualization_fn, i, curr_icp](const auto &p, const auto &q, auto &c) {
      if (visualization_fn)
        (*visualization_fn)(p, q, c, i, curr_icp.T);
    };

    curr_icp = icp(P_source, Q_target, correspondence_fn, minimization_fn, std::make_optional(wrapped_viz), Eigen::Matrix4d::Identity(), icp_duration,
                   iterations);

    // Update the transformation for the current cloud AND all previous clouds
    for (size_t j = 0; j <= i; ++j) {
      current_clouds[j] = transform_vector_points(current_clouds[j], curr_icp.T.block<3, 3>(0, 0), curr_icp.T.block<3, 1>(0, 3));
    }

    global_T = curr_icp.T * global_T;
  }
  return {.T = global_T, .chi = curr_icp.chi};
}
