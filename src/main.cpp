#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include <vector>
#include <iostream>

struct Point3D {
    float x, y, z;
};

int main(int argc, char** argv) {
    polyscope::init();
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    std::vector<Point3D> points = {
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f}, {0.5f, 0.5f, 1.0f}, {0.2f, 0.8f, 0.3f}
    };

    auto* pc = polyscope::registerPointCloud("Source Cloud", points);

    pc->setPointRadius(0.01);
    pc->setPointColor({1.0f, 0.0f, 0.0f}); // RGB

    std::cout << "Polyscope initialized. Press 'q' in the window to exit." << std::endl;

    polyscope::show();

    return 0;
}
