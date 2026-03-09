#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include <iostream>
#include <vector>

struct Point3D {
    float x, y, z;
};

const char *corrMethods[] = {"Naive (Brute Force)", "KD-Tree"};
static int selectedCorr = 0;

const char *miniMethods[] = {"Point-to-Point (SVD)", "Point-to-Point (LS)", "Point-to-Plane (LS)", "Generalized ICP"};
static int selectedMini = 0;

void ICPSettingsCallback() {
    ImGui::Begin("ICP Settings");

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

    std::vector<Point3D> points = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
                                   {1.0f, 1.0f, 0.0f}, {0.5f, 0.5f, 1.0f}, {0.2f, 0.8f, 0.3f}};

    auto *pc = polyscope::registerPointCloud("Source Cloud", points);

    pc->setPointRadius(0.01);
    pc->setPointColor({1.0f, 0.0f, 0.0f}); // RGB

    std::cout << "Polyscope initialized. Press 'q' in the window to exit." << std::endl;

    polyscope::options::openImGuiWindowForUserCallback = false;

    polyscope::state::userCallback = mainCallback;

    polyscope::show();

    return 0;
}
