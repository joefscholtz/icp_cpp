#pragma once
#include <chrono>

using namespace std::chrono_literals;

struct ICPDuration {
  std::chrono::duration<double> correspondence_duration{0s}, minimization_duration{0s}, icp_duration{0s};
};

class Timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<double> duration;

public:
  Timer() { start = std::chrono::high_resolution_clock::now(); }
  void stop_timer() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
  }
  auto get_duration() {
    stop_timer();
    return duration;
  }
};
