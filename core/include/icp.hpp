#pragma once
#include "correspondences.hpp"
#include "math_utils.hpp"
#include "minimization.hpp"

using VisualizationFunctionType =
    std::function<void(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, std::vector<correspondence_t> &correspondences)>;

auto icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
         MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
         std::optional<const Eigen::Matrix4d> T0, std::shared_ptr<ICPDuration> icp_duration, const size_t iterations = 20) -> ICPResult;

auto frame_to_frame_icp(const std::vector<Eigen::Vector3d> &P, const std::vector<Eigen::Vector3d> &Q, CorrespondenceFunctionType &correspondence_fn,
                        MinimizationFunctionType &minimization_fn, std::optional<VisualizationFunctionType> visualization_fn,
                        std::optional<const Eigen::Matrix4d> T0, const size_t iterations) -> ICPResult;
