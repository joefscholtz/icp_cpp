#include "io.hpp"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <Eigen/Core>
#include <iostream>
#include <nfd.hpp>
#include <vector>

const char *corrMethods[] = {"Naive (Brute Force)", "KD-Tree"};
static int selectedCorr = 0;

const char *miniMethods[] = {"Point-to-Point (SVD)", "Point-to-Point (LS)", "Point-to-Plane (LS)", "Generalized ICP"};
static int selectedMini = 0;

void ICPSettingsCallback() {
  ImGui::Begin("ICP");

  if (ImGui::Button("Select Source Mesh")) {
    NFD_Init();

    nfdu8char_t *outPath;
    nfdu8filteritem_t filters[1] = {{"Mesh files", "ply,obj"}};

    nfdresult_t result = NFD_OpenDialog(&outPath, filters, 1, NULL);

    if (result == NFD_OKAY) {
      std::cout << "Selected: " << outPath << std::endl;
      auto points = loadMesh(outPath);
      auto *pc = polyscope::registerPointCloud("Source Cloud", points);

      pc->setPointRadius(0.001);
      pc->setPointColor({1.0f, 0.0f, 0.0f}); // RGB
      //
      NFD_FreePath(outPath); // Clean up
    }

    NFD_Quit();
  }

  ImGui::Text("Pipeline Configuration");
  ImGui::Separator();

  ImGui::Combo("Correspondence", &selectedCorr, corrMethods, IM_ARRAYSIZE(corrMethods));

  ImGui::Combo("Minimization", &selectedMini, miniMethods, IM_ARRAYSIZE(miniMethods));

  ImGui::Separator();

  if (ImGui::Button("Run Iteration")) {
    if (selectedCorr == 0) {
    } else {
    }

    if (selectedMini == 0) {
      std::cout << "Running Point-to-Point with SVD..." << std::endl;
    }
  }

  ImGui::Text("Current RMSE: %.6f", 0.00234);
  ImGui::End();
}

void mainCallback() { ICPSettingsCallback(); }

int main(int argc, char **argv) {
  polyscope::init();
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  polyscope::options::openImGuiWindowForUserCallback = false;

  polyscope::state::userCallback = mainCallback;

  polyscope::show();

  return 0;
}
