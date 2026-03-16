#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstddef>
#include <nanoflann.hpp>
#include <optional>

struct VectorAdaptor {
  const std::vector<Eigen::Vector3d> &pts;
  VectorAdaptor(const std::vector<Eigen::Vector3d> &p) : pts(p) {}
  size_t kdtree_get_point_count() const { return pts.size(); }
  double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx](dim); }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

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

struct PointToPlaneError {
  PointToPlaneError(Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d n_q, std::optional<Eigen::Vector3d> n_p = std::nullopt)
      : p_(p), q_(q), n_q_(n_q), n_p_(n_p) {}

  template <typename T> bool operator()(const T *const angle_axis, const T *const translation, T *residual) const {
    T p[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
    T p_rotated[3];
    ceres::AngleAxisRotatePoint(angle_axis, p, p_rotated);

    // Transformed point p'
    T p_prime[3] = {p_rotated[0] + translation[0], p_rotated[1] + translation[1], p_rotated[2] + translation[2]};

    // Error vector (p' - q)
    T diff[3] = {p_prime[0] - T(q_.x()), p_prime[1] - T(q_.y()), p_prime[2] - T(q_.z())};

    if (n_p_.has_value()) {
      // Symmetric ICP: Project onto the average of the normals
      // Transform source normal n_p into the target frame
      T np[3] = {T(n_p_->x()), T(n_p_->y()), T(n_p_->z())};
      T np_rotated[3];
      ceres::AngleAxisRotatePoint(angle_axis, np, np_rotated);

      T n_avg[3] = {(np_rotated[0] + T(n_q_.x())) * T(0.5), (np_rotated[1] + T(n_q_.y())) * T(0.5), (np_rotated[2] + T(n_q_.z())) * T(0.5)};

      // Normalize average normal
      T norm = ceres::sqrt(n_avg[0] * n_avg[0] + n_avg[1] * n_avg[1] + n_avg[2] * n_avg[2] + T(1e-10));
      residual[0] = (diff[0] * n_avg[0] + diff[1] * n_avg[1] + diff[2] * n_avg[2]) / norm;
    } else {
      // Standard Point-to-Plane: Project onto target normal n_q
      residual[0] = diff[0] * T(n_q_.x()) + diff[1] * T(n_q_.y()) + diff[2] * T(n_q_.z());
    }

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &p, const Eigen::Vector3d &q, const Eigen::Vector3d &n_q,
                                     std::optional<Eigen::Vector3d> n_p) {
    return new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 3, 3>(new PointToPlaneError(p, q, n_q, n_p));
  }

  const Eigen::Vector3d p_, q_, n_q_;
  const std::optional<Eigen::Vector3d> n_p_;
};

inline auto transform_vector_points(const std::vector<Eigen::Vector3d> &P, Eigen::Matrix3d R, Eigen::Vector3d t) -> std::vector<Eigen::Vector3d> {
  size_t N = P.size();
  std::vector<Eigen::Vector3d> P_transformed(N);
  for (size_t i = 0; i < N; i++) {
    P_transformed[i] = R * P[i] + t;
  }
  return P_transformed;
}

inline auto estimate_normals(const std::vector<Eigen::Vector3d> &points, int k = 10) -> std::vector<Eigen::Vector3d> {
  std::vector<Eigen::Vector3d> normals(points.size());

  // Setup nanoflann adaptor for the cloud
  VectorAdaptor adaptor(points);

  using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, VectorAdaptor>, VectorAdaptor, 3>;

  my_kd_tree_t index(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  for (size_t i = 0; i < points.size(); ++i) {
    std::vector<size_t> ret_indices(k);
    std::vector<double> out_dist_sq(k);
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(ret_indices.data(), out_dist_sq.data());
    index.findNeighbors(resultSet, points[i].data(), nanoflann::SearchParameters());

    // Compute local covariance
    Eigen::Vector3d centroid(0, 0, 0);
    for (size_t idx : ret_indices)
      centroid += points[idx];
    centroid /= static_cast<double>(k);

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (size_t idx : ret_indices) {
      Eigen::Vector3d centered = points[idx] - centroid;
      covariance += centered * centered.transpose();
    }

    // Smallest eigenvector is the normal
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    normals[i] = solver.eigenvectors().col(0).normalized();

    // Ensure consistent orientation (pointing towards origin or "up")
    if (normals[i].dot(points[i]) > 0)
      normals[i] *= -1.0;
  }
  return normals;
}

inline auto calculate_cloud_extent(const std::vector<Eigen::Vector3d> &points) -> double {
  if (points.empty())
    return 0.0;
  Eigen::Vector3d min_p = points[0];
  Eigen::Vector3d max_p = points[0];
  for (const auto &p : points) {
    min_p = min_p.cwiseMin(p);
    max_p = max_p.cwiseMax(p);
  }
  return (max_p - min_p).norm(); // Diagonal of the AABB
}

struct VoxelKey {
  long x, y, z;
  bool operator<(const VoxelKey &other) const {
    if (x != other.x)
      return x < other.x;
    if (y != other.y)
      return y < other.y;
    return z < other.z;
  }
};

struct VoxelData {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  size_t count = 0;
};

inline auto voxel_downsample(const std::vector<Eigen::Vector3d> &points, double voxel_size) -> std::vector<Eigen::Vector3d> {
  if (voxel_size <= 0.0)
    return points;

  std::map<VoxelKey, VoxelData> grid;
  for (const auto &p : points) {
    VoxelKey key{static_cast<long>(std::floor(p.x() / voxel_size)), static_cast<long>(std::floor(p.y() / voxel_size)),
                 static_cast<long>(std::floor(p.z() / voxel_size))};
    grid[key].sum += p;
    grid[key].count++;
  }

  std::vector<Eigen::Vector3d> downsampled;
  downsampled.reserve(grid.size());
  for (auto const &[key, data] : grid) {
    downsampled.push_back(data.sum / static_cast<double>(data.count));
  }
  return downsampled;
}
