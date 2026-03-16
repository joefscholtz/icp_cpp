#include "icp.hpp"
#include "io.hpp"
#include "minimization.hpp"
#include "polyscope/point_cloud.h"
#include "polyscope/point_cloud_vector_quantity.h"
#include "polyscope/polyscope.h"
#include "time_utils.hpp"
#include <Eigen/Core>
#include <atomic>
#include <cstddef>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <nfd.h>
#include <nfd.hpp>
#include <optional>
#include <thread>
#include <vector>

enum class CorrespondenceType { Naive, KDTree, KDTreeNanoflann };
enum class MinimizationType { PointToPointSVD, PointToPointLS, PointToPlaneLS, GeneralizedICP };

struct CorrespondenceOption {
  CorrespondenceType type;
  std::string name;
  CorrespondenceFunctionType function;
};

struct MinimizationOption {
  MinimizationType type;
  std::string name;
  MinimizationFunctionType function;
};

const std::vector<CorrespondenceOption> correspondence_registry = {
    {CorrespondenceType::Naive, "Naive Nearest Neighbor (Brute Force)", correspondence_nn},
    {CorrespondenceType::KDTree, "KDTree", correspondence_kdtree},
    {CorrespondenceType::KDTreeNanoflann, "KDTree (using nanoflann)", correspondence_kdtree_nanoflann}};

const std::vector<MinimizationOption> minimization_registry = {
    {MinimizationType::PointToPointSVD, "Point-to-Point (SVD)", minimize_point_to_point_svd},
    {MinimizationType::PointToPointLS, "Point-to-Point (Non Linear Least Squares)", minimize_point_to_point_ls},
    {MinimizationType::PointToPlaneLS, "Point-to-Plane (Non Linear Least Squares)", minimize_point_to_plane_ls},
    {MinimizationType::GeneralizedICP, "Generalized ICP", minimize_generalized_icp}};

struct AppState {
  std::shared_ptr<ICPDuration> icp_duration = std::make_shared<ICPDuration>();
  ICPResult icp_res;

  CorrespondenceFunctionType correspondence_fn;
  MinimizationFunctionType minimization_fn;

  std::vector<std::vector<Eigen::Vector3d>> point_clouds;
  std::vector<std::string> cloud_names;
  std::vector<Eigen::Vector3d> points;
  size_t P_idx = 0;
  size_t Q_idx = 0;

  int icp_iterations = 100;
  int num_scales = 1;
  bool stop_at_iter = true;
  bool start_transform_with_last_estimate = false;
  int viz_pause_ms = 100;
  bool cumulative_icp = false;

  CorrespondenceType selected_corr = CorrespondenceType::KDTreeNanoflann;
  MinimizationType selected_min = MinimizationType::PointToPointSVD;

  std::future<ICPResult> icp_future;
  std::atomic<bool> is_running{false};
  std::mutex viz_mutex;

  std::vector<Eigen::Vector3d> viz_P;
  std::vector<Eigen::Vector3d> viz_Q;
  std::vector<correspondence_t> viz_correspondences;
  std::atomic<bool> has_new_viz_data{false};

  std::atomic<bool> stop_requested{false};
};

void ICPSettingsCallback(AppState &state) {
  const char *corr_methods[] = {"Naive Nearest Neighbor (Brute Force)", "KD-Tree"};
  const char *min_methods[] = {"Point-to-Point (SVD)", "Point-to-Point (LS)", "Point-to-Plane (LS)", "Generalized ICP"};

  ImGui::Begin("Point Cloud Management");

  if (ImGui::Button("Select Meshes (Min 2)") && !state.is_running) {
    NFD_Init();
    const nfdpathset_t *path_set;
    nfdu8filteritem_t filters[1] = {{"Mesh files", "ply,obj"}};
    nfdresult_t ndf_result = NFD_OpenDialogMultiple(&path_set, filters, 1, NULL);

    if (ndf_result == NFD_OKAY) {
      for (const auto &name : state.cloud_names) {
        polyscope::removePointCloud(name, false); // false = don't error if not found
      }
      state.point_clouds.clear();
      state.cloud_names.clear();
      state.points.clear();

      nfdpathsetsize_t count;
      NFD_PathSet_GetCount(path_set, &count);
      if (count >= 2) {
        state.point_clouds.clear();
        state.cloud_names.clear();
        for (nfdpathsetsize_t i = 0; i < count; ++i) {
          nfdu8char_t *path;
          NFD_PathSet_GetPath(path_set, i, &path);

          std::string full_path(path);
          size_t last_slash = full_path.find_last_of("/\\");
          size_t last_dot = full_path.find_last_of(".");
          std::string name = (last_slash == std::string::npos) ? full_path : full_path.substr(last_slash + 1);
          if (last_dot != std::string::npos && last_dot > last_slash) {
            name = name.substr(0, name.find_last_of("."));
          }
          state.cloud_names.push_back(name);

          state.points = loadMesh(path);
          state.point_clouds.push_back(state.points);

          auto *pc = polyscope::registerPointCloud(name, state.points);
          if (i == 0)
            pc->setPointColor({1.0f, 0.0f, 0.0f}); // red
          else
            pc->setPointColor({0.0f, 1.0f, 0.0f}); // green
          pc->setMaterial("clay");

          NFD_PathSet_FreePath(path);
        }
      }
      NFD_PathSet_Free(path_set);
    }
    NFD_Quit();
  }

  if (!state.point_clouds.empty() && !state.is_running) {
    ImGui::SameLine();
    if (ImGui::Button("Reverse Order")) {
      std::reverse(state.point_clouds.begin(), state.point_clouds.end());
      std::reverse(state.cloud_names.begin(), state.cloud_names.end());

      // Update Polyscope colors/names
      for (size_t j = 0; j < state.point_clouds.size(); ++j) {
        auto *pc = polyscope::registerPointCloud(state.cloud_names[j], state.point_clouds[j]);
        if (j == 0)
          pc->setPointColor({1.0f, 0.0f, 0.0f}); // red
        else
          pc->setPointColor({0.0f, 1.0f, 0.0f}); // green
      }
    }

    ImGui::Separator();
    ImGui::Text("Drag and Drop to Reorder:");
    for (int i = 0; i < (int)state.point_clouds.size(); i++) {
      std::string label = std::to_string(i) + ": " + state.cloud_names[i];
      ImGui::Selectable(label.c_str());

      if (ImGui::BeginDragDropSource()) {
        ImGui::SetDragDropPayload("DND_CLOUD", &i, sizeof(int));
        ImGui::Text("Moving %s", state.cloud_names[i].c_str());
        ImGui::EndDragDropSource();
      }

      if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("DND_CLOUD")) {
          int source_idx = *(const int *)payload->Data;
          if (source_idx != i) {
            auto it_pc = state.point_clouds.begin();
            auto it_name = state.cloud_names.begin();

            auto pc_val = state.point_clouds[source_idx];
            auto name_val = state.cloud_names[source_idx];

            state.point_clouds.erase(it_pc + source_idx);
            state.cloud_names.erase(it_name + source_idx);

            state.point_clouds.insert(state.point_clouds.begin() + i, pc_val);
            state.cloud_names.insert(state.cloud_names.begin() + i, name_val);

            for (size_t j = 0; j < state.point_clouds.size(); ++j) {
              auto *pc = polyscope::registerPointCloud(state.cloud_names[j], state.point_clouds[j]);
              if (j == 0)
                pc->setPointColor({1.0f, 0.0f, 0.0f}); // red
              else
                pc->setPointColor({0.0f, 1.0f, 0.0f}); // green
            }
          }
        }
        ImGui::EndDragDropTarget();
      }
    }
  }
  ImGui::End();

  ImGui::Begin("ICP");
  ImGui::Text("ICP Configuration");
  ImGui::Separator();
  ImGui::InputInt("Iterations", &state.icp_iterations);

  ImGui::SliderInt("Multi-Scale Downsample\nLevels", &state.num_scales, 1, 500);

  ImGui::Checkbox("Cumulative (Scan-to-Map)", &state.cumulative_icp);

  ImGui::Checkbox("Stop at each iteration", &state.stop_at_iter);
  if (state.stop_at_iter) {
    ImGui::Indent();
    ImGui::InputInt("Pause (ms)", &state.viz_pause_ms);
    ImGui::Unindent();
  }
  ImGui::Checkbox("Use last frame transform iteration \nas first estimation for the next", &state.start_transform_with_last_estimate);

  std::string current_corr_label = "None";
  for (const auto &opt : correspondence_registry) {
    if (opt.type == state.selected_corr)
      current_corr_label = opt.name;
  }

  if (ImGui::BeginCombo("Correspondence", current_corr_label.c_str())) {
    for (const auto &opt : correspondence_registry) {
      bool is_selected = (state.selected_corr == opt.type);
      if (ImGui::Selectable(opt.name.c_str(), is_selected)) {
        state.selected_corr = opt.type;
      }
      if (is_selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  std::string current_min_label = "None";
  for (const auto &opt : minimization_registry) {
    if (opt.type == state.selected_min)
      current_min_label = opt.name;
  }

  if (ImGui::BeginCombo("Minimization", current_min_label.c_str())) {
    for (const auto &opt : minimization_registry) {
      bool is_selected = (state.selected_min == opt.type);
      if (ImGui::Selectable(opt.name.c_str(), is_selected)) {
        state.selected_min = opt.type;
      }
      if (is_selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::Separator();

  if (state.is_running) {
    if (state.has_new_viz_data) {
      std::lock_guard<std::mutex> lock(state.viz_mutex);

      for (size_t i = 0; i < state.point_clouds.size(); ++i) {
        auto *pc = polyscope::getPointCloud(state.cloud_names[i]);
        if (!pc)
          continue;

        pc->removeQuantity("Matching Vectors");

        if (i < state.P_idx && i < state.Q_idx) {
          pc->updatePointPositions(state.point_clouds[i]);
          pc->setPointColor({0.0f, 1.0f, 0.0f}); // green
        } else if (i == state.P_idx) {
          pc->updatePointPositions(state.viz_P);
          pc->setPointColor({0.2f, 0.8f, 0.2f}); // pale green

          if (!state.viz_correspondences.empty()) {
            std::vector<glm::vec3> vectors;
            for (auto &c : state.viz_correspondences) {
              Eigen::Vector3d diff = state.viz_Q[c.second] - state.viz_P[c.first];
              vectors.emplace_back((float)diff.x(), (float)diff.y(), (float)diff.z());
            }
            auto *matching_vecs = pc->addVectorQuantity("Matching Vectors", vectors, polyscope::VectorType::AMBIENT);
            matching_vecs->setVectorColor({0.0f, 0.0f, 1.0f});
            matching_vecs->setVectorLengthScale(1.0, false);
            matching_vecs->setVectorRadius(0.0005, false);
            matching_vecs->setEnabled(true);
          }
        } else if (i == state.Q_idx) {
          pc->setPointColor({1.0f, 0.0f, 0.0f}); // red
        } else {
          pc->setPointColor({0.5f, 0.5f, 0.5f}); // grey
        }
      }
      state.has_new_viz_data = false;
    }
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button("STOP ICP", ImVec2(-1, 0))) {
      state.stop_requested = true;
    }
    ImGui::PopStyleColor();

    float t = (float)ImGui::GetTime();
    const char *frames[] = {"[ .     ]", "[ ..    ]", "[ ...   ]", "[  ...  ]", "[   ..  ]", "[    .  ]", "[     . ]"};
    ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Running ICP %s", frames[(int)(t * 10.0f) % 7]);

    if (state.icp_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      state.icp_res = state.icp_future.get();
      state.is_running = false;
      state.stop_requested = false; // Reset for next run
      for (size_t i = 0; i < state.point_clouds.size(); ++i) {
        auto *pc = polyscope::getPointCloud(state.cloud_names[i]);
        if (pc)
          pc->removeQuantity("Matching Vectors");
        pc->updatePointPositions(state.point_clouds[i]);
      }
    }
  } else {
    if (ImGui::Button("Run ICP") && state.point_clouds.size() > 1) {
      bool corr_was_selected{false}, min_was_selected{false};
      for (const auto &opt : correspondence_registry) {
        if (opt.type == state.selected_corr) {
          state.correspondence_fn = opt.function;
          corr_was_selected = true;
        }
      }
      for (const auto &opt : minimization_registry) {
        if (opt.type == state.selected_min) {
          state.minimization_fn = opt.function;
          min_was_selected = true;
        }
      }

      if (corr_was_selected && min_was_selected) {
        std::optional<VisualizationFunctionType> viz_callback = std::nullopt;
        if (state.stop_at_iter) {
          viz_callback = [&](const auto &P, const auto &Q, auto &corr, size_t P_idx, size_t Q_idx) {
            {
              std::lock_guard<std::mutex> lock(state.viz_mutex);
              state.viz_P = P;
              state.viz_Q = Q;
              state.viz_correspondences = corr;
              state.P_idx = P_idx;
              state.Q_idx = Q_idx;
              state.has_new_viz_data = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(state.viz_pause_ms));
          };
        }

        state.is_running = true;
        state.stop_requested = false;
        state.icp_future = std::async(std::launch::async, [&state, viz_callback]() {
          return frame_to_frame_icp(state.point_clouds, state.correspondence_fn, state.minimization_fn, viz_callback, std::nullopt,
                                    state.start_transform_with_last_estimate, state.icp_duration, state.icp_iterations, state.num_scales,
                                    state.cumulative_icp, state.stop_requested);
        });
      }
    }
  }

  if (state.icp_duration && state.icp_duration->icp_duration.count() > 0) {
    ImGui::Text("Correspondence: %.3f ms", state.icp_duration->correspondence_duration.count() * 1000.0);
    ImGui::Text("Minimization: %.3f ms", state.icp_duration->minimization_duration.count() * 1000.0);
  }

  ImGui::End();
}

int main(int argc, char **argv) {
  AppState state;
  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
  polyscope::options::openImGuiWindowForUserCallback = false;
  polyscope::state::userCallback = [&state]() { ICPSettingsCallback(state); };
  polyscope::show();
  return 0;
}
