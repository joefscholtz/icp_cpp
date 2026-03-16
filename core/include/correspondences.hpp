#pragma once
#include "kdtree.hpp"
#include "math_utils.hpp"
#include "time_utils.hpp"
#include <functional>
#include <limits>
#include <nanoflann.hpp>
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
  std::optional<double> max_dist_sq;
  if (max_dist)
    max_dist_sq = (*max_dist) * (*max_dist);

  for (size_t i = 0; i < P.size(); ++i) {
    double d_min_sq = std::numeric_limits<double>::max();
    size_t best_idx = 0;

    for (size_t j = 0; j < Q.size(); ++j) {
      double d2 = (P[i] - Q[j]).squaredNorm();
      if (d2 < d_min_sq) {
        d_min_sq = d2;
        best_idx = j;
      }
    }

    if (!max_dist_sq.has_value() || d_min_sq < *max_dist_sq) {
      correspondences.emplace_back(i, best_idx);
    }
  }
  if (duration_ptr != nullptr) {
    *duration_ptr = timer.get_duration();
  }
  return correspondences;
}
inline auto correspondence_kdtree(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                  std::shared_ptr<std::chrono::duration<double>> duration_ptr, std::optional<double> max_dist)
    -> std::vector<correspondence_t> {
  std::vector<correspondence_t> correspondences;

  Timer timer;
  KDTree kdtree(Q);
  std::optional<double> max_dist_sq;
  if (max_dist)
    max_dist_sq = (*max_dist) * (*max_dist);

  for (size_t i = 0; i < P.size(); ++i) {

    size_t best_idx = kdtree.find_nearest_idx(P[i]);
    double d_min_sq = (P[i] - Q[best_idx]).squaredNorm();

    if (!max_dist_sq.has_value() || d_min_sq < *max_dist_sq) {
      correspondences.emplace_back(i, best_idx);
    }
  }
  if (duration_ptr != nullptr) {
    *duration_ptr = timer.get_duration();
  }
  return correspondences;
}

inline auto correspondence_kdtree_nanoflann(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q,
                                            std::shared_ptr<std::chrono::duration<double>> duration_ptr, std::optional<double> max_dist)
    -> std::vector<correspondence_t> {

  std::vector<correspondence_t> correspondences;
  Timer timer;

  if (Q.empty())
    return correspondences;

  VectorAdaptor adaptor(Q);

  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, VectorAdaptor>, VectorAdaptor, 3>;

  my_kd_tree_t index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  double max_dist_sq = max_dist ? (*max_dist) * (*max_dist) : std::numeric_limits<double>::max();

  for (size_t i = 0; i < P.size(); ++i) {
    size_t ret_index;
    double out_dist_sq;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &out_dist_sq);

    index.findNeighbors(resultSet, P[i].data(), nanoflann::SearchParameters());

    if (out_dist_sq < max_dist_sq) {
      correspondences.emplace_back(i, ret_index);
    }
  }

  if (duration_ptr != nullptr) {
    *duration_ptr = timer.get_duration();
  }

  return correspondences;
}
