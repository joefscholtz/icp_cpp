#include "icp.hpp"
#include "io.hpp"
#include "minimization.hpp"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "time_utils.hpp"
#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <nfd.h>
#include <nfd.hpp>
#include <optional>
#include <vector>

struct AppState {
  std::shared_ptr<ICPDuration> icp_duration = std::make_shared<ICPDuration>();
  ICPResult icp_res;

  CorrespondenceFunctionType correspondence_fn;
  MinimizationFunctionType minimization_fn;

  std::vector<std::vector<Eigen::Vector3d>> point_clouds;
  std::vector<Eigen::Vector3d> points;

  int icp_iterations = 10;
  bool stop_at_iter = false;

  int selectedCorr = 0;
  int selectedMini = 0;
};

void ICPSettingsCallback(AppState &state) {
  const char *corrMethods[] = {"Naive (Brute Force)", "KD-Tree"};
  const char *miniMethods[] = {"Point-to-Point (SVD)", "Point-to-Point (LS)", "Point-to-Plane (LS)", "Generalized ICP"};

  ImGui::Begin("ICP");

  if (ImGui::Button("Select Meshes (Min 2)")) {
    NFD_Init();

    const nfdpathset_t *pathSet;
    nfdu8filteritem_t filters[1] = {{"Mesh files", "ply,obj"}};

    nfdresult_t result = NFD_OpenDialogMultiple(&pathSet, filters, 1, NULL);

    if (result == NFD_OKAY) {
      nfdpathsetsize_t count;
      NFD_PathSet_GetCount(pathSet, &count);

      if (count < 2) {
        std::cerr << "Please select at least two files!" << std::endl;
      } else {
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

  ImGui::Checkbox("Stop at each iteration", &state.stop_at_iter);

  ImGui::Combo("Correspondence", &state.selectedCorr, corrMethods, IM_ARRAYSIZE(corrMethods));

  ImGui::Combo("Minimization", &state.selectedMini, miniMethods, IM_ARRAYSIZE(miniMethods));

  ImGui::Separator();

  if (ImGui::Button("Run Iteration")) {
    if (state.selectedCorr == 0) {
      state.correspondence_fn = correspondence_nn;
    } else {
    }

    if (state.selectedMini == 0) {
      state.minimization_fn = minimize_point_to_point_svd;
    }
    if (state.point_clouds.size() > 1) {
      state.icp_res = icp(state.point_clouds[0], state.point_clouds[1], state.correspondence_fn, state.minimization_fn, std::nullopt, std::nullopt,
                          state.icp_duration, state.icp_iterations);
      // icp_res = icp(point_clouds[0], point_clouds[1], correspondence_fn, minimization_fn, std::nullopt, std::nullopt, nullptr, icp_iterations);
      auto P_curr = transform_vector_points(state.point_clouds[0], state.icp_res.T.block<3, 3>(0, 0), state.icp_res.T.block<3, 1>(0, 3));
      polyscope::registerPointCloud("Source Cloud", P_curr);
    }
  }

  ImGui::Text("Correspondence function mean duration %.3f ms", state.icp_duration->correspondence_duration.count() * 1000.0);
  ImGui::Text("Minimization function mean duration %.3f ms", state.icp_duration->minimization_duration.count() * 1000.0);
  // ImGui::Text("Current RMSE: %.6f", icp_res.chi);
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
