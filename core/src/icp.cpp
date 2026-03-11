#include "correspondences.hpp"
#include "math_utils.hpp"
#include "minimization.hpp"
#include "time_utils.hpp"
#include <iostream>
#include <memory>

using VisualizationFunctionType =
    std::function<void(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, std::vector<correspondence_t> &correspondences)>;

auto icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
         MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
         std::optional<const Eigen::Matrix4d> T0, std::shared_ptr<ICPDuration> icp_duration, const size_t iterations = 20) -> ICPResult {
  // std::cout << "Running icp" << std::endl;
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

    // std::cout << "Computing Transform" << std::endl;
    // std::cout << "T:" << T << std::endl;
    // std::cout << "min_result.T:" << min_result.T << std::endl;

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

  // std::cout << "End icp" << std::endl;

  return {.T = T, .chi = min_result.chi};
}

auto frame_to_frame_icp(std::vector<std::vector<Eigen::Vector3d>> &point_clouds, CorrespondenceFunctionType &correspondence_fn,
                        MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
                        std::optional<const Eigen::Matrix4d> T0, std::shared_ptr<ICPDuration> icp_duration, const size_t iterations = 20)
    -> ICPResult {

  ICPResult curr_icp;
  Eigen::Matrix4d T = T0.value_or(Eigen::Matrix4d::Identity());
  for (size_t i = 0; i < point_clouds.size() - 1; i++) {
    auto P = point_clouds[i];
    auto Q = point_clouds[i + 1];
    curr_icp = icp(P, Q, correspondence_fn, minimization_fn, visualization_fn, T, icp_duration, iterations);
    T = curr_icp.T;
  }
  return curr_icp;
}
