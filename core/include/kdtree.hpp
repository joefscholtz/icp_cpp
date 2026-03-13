#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <memory>

class KDNode {
public:
  KDNode(std::weak_ptr<KDNode> parent_node, std::shared_ptr<KDNode> first_child, std::shared_ptr<KDNode> second_child, size_t depth,
         std::shared_ptr<Eigen::Vector3d> data, size_t index)
      : _parent_node(parent_node), _first_child(first_child), _second_child(second_child), _depth(depth), _data(data), _index(index) {}

  KDNode(std::shared_ptr<Eigen::Vector3d> data, size_t index) : KDNode(std::weak_ptr<KDNode>(), nullptr, nullptr, 0, data, index) {}

  inline auto get_parent() -> std::weak_ptr<KDNode> { return _parent_node; }
  inline void set_parent(std::weak_ptr<KDNode> parent) { _parent_node = parent; }

  inline auto get_first() -> std::shared_ptr<KDNode> { return _first_child; }
  inline void set_first(std::shared_ptr<KDNode> first) { _first_child = first; }

  inline auto get_second() -> std::shared_ptr<KDNode> { return _second_child; }
  inline void set_second(std::shared_ptr<KDNode> second) { _second_child = second; }

  inline auto get_depth() -> size_t { return _depth; }
  inline void set_depth(size_t depth) { _depth = depth; }

  inline auto get_data() -> std::shared_ptr<Eigen::Vector3d> { return _data; }
  inline void set_data(std::shared_ptr<Eigen::Vector3d> &data) { _data = data; }

  inline auto get_index() -> size_t { return _index; }
  inline void set_index(size_t index) { _index = index; }

  inline auto can_be_first_of(KDNode &other) -> bool { return (*_data)(other.get_depth() % 3) < (*(other.get_data()))(other.get_depth() % 3); }

  inline auto get_distance(KDNode &other) -> double { return ((*_data) - (*(other.get_data()))).norm(); }

  inline auto get_perpendicular_distance_from(KDNode &other) -> double {
    double signed_distance = (*_data)(other.get_depth() % 3) - (*(other.get_data()))(other.get_depth() % 3);
    double abs_distance = std::abs(signed_distance);
    return abs_distance;
  }

private:
  std::weak_ptr<KDNode> _parent_node;
  std::shared_ptr<KDNode> _first_child, _second_child;
  size_t _depth;
  std::shared_ptr<Eigen::Vector3d> _data;
  size_t _index;
};

class KDTree {
public:
  KDTree(std::vector<Eigen::Vector3d> &point_cloud) { make_tree(point_cloud); }

  auto find_nearest_idx(Eigen::Vector3d &point) -> size_t {
    auto query_node = std::make_shared<KDNode>(std::make_shared<Eigen::Vector3d>(point), 0);
    std::shared_ptr<KDNode> best_node = nullptr;
    double best_dist = std::numeric_limits<double>::max();

    search_recursive(_root, query_node, best_node, best_dist);

    return best_node ? best_node->get_index() : 0;
  }

private:
  void search_recursive(std::shared_ptr<KDNode> current, std::shared_ptr<KDNode> query, std::shared_ptr<KDNode> &best_node, double &best_dist) {
    if (!current)
      return;

    // 1. Check current node
    double d = query->get_distance(*current);
    if (d < best_dist) {
      best_dist = d;
      best_node = current;
    }

    // 2. Decide which way to go first
    bool query_is_first = query->can_be_first_of(*current);
    auto near_child = query_is_first ? current->get_first() : current->get_second();
    auto far_child = query_is_first ? current->get_second() : current->get_first();

    // 3. Search the "near" side
    search_recursive(near_child, query, best_node, best_dist);

    // 4. Check if we need to search the "far" side
    double plane_dist = query->get_perpendicular_distance_from(*current);

    if (plane_dist < best_dist) {
      search_recursive(far_child, query, best_node, best_dist);
    }
  }

  inline auto find_parent(std::shared_ptr<KDNode> new_node, std::shared_ptr<KDNode> current_node, bool insert = false) -> std::shared_ptr<KDNode> {
    if (new_node->can_be_first_of(*current_node)) {
      if (current_node->get_first() == nullptr) {
        if (insert) {
          new_node->set_parent(current_node);
          new_node->set_depth(current_node->get_depth() + 1);
          current_node->set_first(new_node);
        }
        return current_node;
      }
      return find_parent(new_node, current_node->get_first(), insert);
    } else {
      if (current_node->get_second() == nullptr) {
        if (insert) {
          new_node->set_parent(current_node);
          new_node->set_depth(current_node->get_depth() + 1);
          current_node->set_second(new_node);
        }
        return current_node;
      }
      return find_parent(new_node, current_node->get_second(), insert);
    }
  }

  inline void make_tree(std::vector<Eigen::Vector3d> &point_cloud) {
    if (point_cloud.empty())
      return;

    _root = std::make_shared<KDNode>(KDNode(std::make_shared<Eigen::Vector3d>(point_cloud[0]), 0));

    for (size_t i = 1; i < point_cloud.size(); i++) {
      auto new_node = std::make_shared<KDNode>(KDNode(std::make_shared<Eigen::Vector3d>(point_cloud[i]), i));
      auto current_node = _root;
      find_parent(new_node, current_node, true);
    }
  }

  std::shared_ptr<KDNode> _root;
};
