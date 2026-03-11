#pragma once
#include "math_utils.hpp"
#include "time_utils.hpp"
#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

using correspondence_t = std::pair<size_t, size_t>;

using CorrespondenceFunctionType =
    std::function<std::vector<correspondence_t>(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                                std::shared_ptr<std::chrono::duration<double>> duration_ptr, std::optional<double> max_dist)>;

inline auto correspondence_nn(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                              std::shared_ptr<std::chrono::duration<double>> duration_ptr, std::optional<double> max_dist)
    -> std::vector<correspondence_t> {
  std::vector<correspondence_t> correspondences;

  Timer timer;
  // Pre-calculating the squared distance threshold if max_dist is provided
  std::optional<double> max_dist_sq;
  if (max_dist)
    max_dist_sq = (*max_dist) * (*max_dist);

  for (size_t i = 0; i < P.size(); ++i) {
    double d_min_sq = std::numeric_limits<double>::max();
    size_t best_idx = 0;
    bool found = false;

    for (size_t j = 0; j < Q.size(); ++j) {
      double d2 = (P[i] - Q[j]).squaredNorm();
      if (d2 < d_min_sq) {
        d_min_sq = d2;
        best_idx = j;
        found = true;
      }
    }

    if (found && (!max_dist_sq.has_value() || d_min_sq < *max_dist_sq)) {
      correspondences.emplace_back(i, best_idx);
    }
  }
  if (duration_ptr != nullptr) {
    *duration_ptr = timer.get_duration();
  }
  return correspondences;
}
