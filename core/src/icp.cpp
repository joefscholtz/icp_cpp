#include "icp.hpp"
#include "correspondences.hpp"
#include "math_utils.hpp"
#include "minimization.hpp"

auto icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
         MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
         std::optional<const Eigen::Matrix4d> T0, const size_t iterations = 20) -> ICPResult {
  Eigen::Matrix4d T = T0.value_or(Eigen::Matrix4d::Identity());

  std::vector<Eigen::Vector3d> P_curr;
  std::vector<correspondence_t> correspondences;
  ICPResult min_result;

  for (size_t i = 0; i < iterations; i++) {
    P_curr = transform_vector_points(P, T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));

    correspondences = correspondence_fn(P_curr, Q, std::nullopt);
    min_result = minimization_fn(P_curr, Q, correspondences);

    T = min_result.T * T;

    if (visualization_fn) {
      (*visualization_fn)(P_curr, Q, correspondences);
    }
  }

  return {.T = T, .chi = min_result.chi};
}

auto frame_to_frame_icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
                        MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
                        std::optional<const Eigen::Matrix4d> T0, const size_t iterations = 20) -> ICPResult {}
