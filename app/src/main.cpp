#include "icp.hpp"
#include "io.hpp"
#include "minimization.hpp"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "time_utils.hpp"
#include <Eigen/Core>
#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <nfd.h>
#include <nfd.hpp>
#include <optional>
#include <thread>
#include <vector>

struct AppState {
  std::shared_ptr<ICPDuration> icp_duration = std::make_shared<ICPDuration>();
  ICPResult icp_res;

  CorrespondenceFunctionType correspondence_fn;
  MinimizationFunctionType minimization_fn;

  std::vector<std::vector<Eigen::Vector3d>> point_clouds;
  std::vector<Eigen::Vector3d> points;
  size_t current_pair_idx = 0;

  int icp_iterations = 10;
  bool stop_at_iter = false;
  int viz_pause_ms = 500;

  int selectedCorr = 0;
  int selectedMini = 0;

  std::future<ICPResult> icp_future;
  std::atomic<bool> is_running{false};
  std::mutex viz_mutex;

  std::vector<Eigen::Vector3d> viz_P;
  std::vector<correspondence_t> viz_correspondences;
  std::atomic<bool> has_new_viz_data{false};

  bool cumulative_icp = false;
  std::vector<std::vector<Eigen::Vector3d>> viz_all_clouds;

  Eigen::Matrix4d current_viz_T = Eigen::Matrix4d::Identity(); // The delta T from the current ICP iteration
  std::vector<Eigen::Matrix4d> cloud_poses;                    // Store the "settled" pose of each cloud
};

void ICPSettingsCallback(AppState &state) {
  const char *corrMethods[] = {"Naive (Brute Force)", "KD-Tree"};
  const char *miniMethods[] = {"Point-to-Point (SVD)", "Point-to-Point (LS)", "Point-to-Plane (LS)", "Generalized ICP"};

  ImGui::Begin("ICP");

  if (ImGui::Button("Select Meshes (Min 2)") && !state.is_running) {
    NFD_Init();
    const nfdpathset_t *pathSet;
    nfdu8filteritem_t filters[1] = {{"Mesh files", "ply,obj"}};
    nfdresult_t result = NFD_OpenDialogMultiple(&pathSet, filters, 1, NULL);

    if (result == NFD_OKAY) {
      nfdpathsetsize_t count;
      NFD_PathSet_GetCount(pathSet, &count);
      if (count >= 2) {
        state.point_clouds.clear();
        for (nfdpathsetsize_t i = 0; i < count; ++i) {
          nfdu8char_t *path;
          NFD_PathSet_GetPath(pathSet, i, &path);
          state.points = loadMesh(path);
          state.point_clouds.push_back(state.points);

          std::string cloudName = (i == 0) ? "Source Cloud" : "Target Cloud " + std::to_string(i);
          auto *pc = polyscope::registerPointCloud(cloudName, state.points);
          if (i == 0)
            pc->setPointColor({1.0f, 0.0f, 0.0f});
          else
            pc->setPointColor({0.0f, 1.0f, 0.0f});
          pc->setMaterial("candy");

          NFD_PathSet_FreePath(path);
        }
      }
      NFD_PathSet_Free(pathSet);
    }
    NFD_Quit();
  }

  ImGui::Text("ICP Configuration");
  ImGui::Separator();
  ImGui::InputInt("Iterations", &state.icp_iterations);

  ImGui::Checkbox("Cumulative (Scan-to-Map)", &state.cumulative_icp);

  ImGui::Checkbox("Stop at each iteration", &state.stop_at_iter);
  if (state.stop_at_iter) {
    ImGui::Indent();
    ImGui::InputInt("Pause (ms)", &state.viz_pause_ms);
    ImGui::Unindent();
  }

  ImGui::Combo("Correspondence", &state.selectedCorr, corrMethods, IM_ARRAYSIZE(corrMethods));
  ImGui::Combo("Minimization", &state.selectedMini, miniMethods, IM_ARRAYSIZE(miniMethods));
  ImGui::Separator();

  if (state.is_running) {
    if (state.has_new_viz_data) {
      std::lock_guard<std::mutex> lock(state.viz_mutex);

      for (size_t i = 0; i < state.point_clouds.size(); ++i) {
        std::string cloudName = (i == 0) ? "Source Cloud" : "Target Cloud " + std::to_string(i);
        auto *pc = polyscope::getPointCloud(cloudName);
        if (!pc)
          continue;

        if (i < state.current_pair_idx) {
          auto P_moved = transform_vector_points(state.point_clouds[i], state.current_viz_T.block<3, 3>(0, 0), state.current_viz_T.block<3, 1>(0, 3));
          pc->updatePointPositions(P_moved);
          pc->setPointColor({0.0f, 1.0f, 0.0f}); // Green

        } else if (i == state.current_pair_idx) {
          pc->updatePointPositions(state.viz_P);
          pc->setPointColor({0.2f, 0.8f, 0.2f}); // Bright Green

          if (!state.viz_correspondences.empty()) {
            std::vector<glm::vec3> vectors;
            for (auto &c : state.viz_correspondences) {
              Eigen::Vector3d diff = state.point_clouds[i + 1][c.second] - state.viz_P[c.first];
              vectors.emplace_back((float)diff.x(), (float)diff.y(), (float)diff.z());
            }
            auto *vQ = pc->addVectorQuantity("Alignment", vectors, polyscope::VectorType::AMBIENT);
            vQ->setVectorColor({0.0f, 0.0f, 1.0f});
            vQ->setVectorLengthScale(1.0, false);
            vQ->setVectorRadius(0.0005, false);
            vQ->setEnabled(true);
          }
        } else if (i == state.current_pair_idx + 1) {
          pc->setPointColor({1.0f, 0.0f, 0.0f}); // Red (Target)
        } else {
          pc->setPointColor({0.5f, 0.5f, 0.5f}); // Grey (Pending)
        }
      }
      state.has_new_viz_data = false;
    }

    float t = (float)ImGui::GetTime();
    const char *frames[] = {"[ .     ]", "[ ..    ]", "[ ...   ]", "[  ...  ]", "[   ..  ]", "[    .  ]", "[     . ]"};
    ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Running ICP %s", frames[(int)(t * 10.0f) % 7]);

    if (state.icp_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      state.icp_res = state.icp_future.get();
      state.is_running = false;
    }
  } else {
    if (ImGui::Button("Run ICP") && state.point_clouds.size() > 1) {
      state.selectedCorr == 0 ? state.correspondence_fn = correspondence_nn : nullptr;
      state.selectedMini == 0 ? state.minimization_fn = minimize_point_to_point_svd : nullptr;

      std::optional<VisualizationFunctionType> viz_callback = std::nullopt;
      if (state.stop_at_iter) {
        int p_ms = state.viz_pause_ms;
        viz_callback = [&](const auto &P, const auto &, auto &corr, size_t pair_idx, const Eigen::Matrix4d &current_T) {
          {
            std::lock_guard<std::mutex> lock(state.viz_mutex);
            state.viz_P = P;
            state.viz_correspondences = corr;
            state.current_pair_idx = pair_idx;
            state.current_viz_T = current_T;
            state.has_new_viz_data = true;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(state.viz_pause_ms));
        };
      }

      state.is_running = true;
      state.icp_future = std::async(std::launch::async, [&state, viz_callback]() {
        return frame_to_frame_icp(state.point_clouds, state.correspondence_fn, state.minimization_fn, viz_callback, std::nullopt, state.icp_duration,
                                  state.icp_iterations);
      });
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
