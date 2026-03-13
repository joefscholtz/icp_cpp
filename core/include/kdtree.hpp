#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <memory>

class KDNode {
public:
  KDNode(std::weak_ptr<KDNode> parent_node, std::shared_ptr<KDNode> first_child, std::shared_ptr<KDNode> second_child,
         std::shared_ptr<Eigen::Vector3d> data, size_t index)
      : _parent_node(parent_node), _first_child(first_child), _second_child(second_child), _data(data), _index(index) {}

  KDNode(std::shared_ptr<Eigen::Vector3d> data, size_t index) : KDNode(std::weak_ptr<KDNode>(), nullptr, nullptr, data, index) {}

  inline auto get_parent() -> std::weak_ptr<KDNode> { return _parent_node; }
  inline void set_parent(std::weak_ptr<KDNode> parent) { _parent_node = parent; }

  inline auto get_first() -> std::shared_ptr<KDNode> { return _first_child; }
  inline void set_first(std::shared_ptr<KDNode> first) { _first_child = first; }

  inline auto get_second() -> std::shared_ptr<KDNode> { return _second_child; }
  inline void set_second(std::shared_ptr<KDNode> second) { _second_child = second; }

  inline auto get_data() -> std::shared_ptr<Eigen::Vector3d> { return _data; }
  inline void set_data(std::shared_ptr<Eigen::Vector3d> &data) { _data = data; }

  inline auto get_index() -> size_t { return _index; }
  inline void set_index(size_t index) { _index = index; }

  inline auto is_first_of(KDNode &other, size_t depth) { return (*_data)(depth % 3) < (*(other.get_data()))(depth % 3); }

private:
  std::weak_ptr<KDNode> _parent_node;
  std::shared_ptr<KDNode> _first_child, _second_child;
  std::shared_ptr<Eigen::Vector3d> _data;
  size_t _index;
};

class KDTree {
public:
  KDTree() = default;

  inline void make_tree(std::vector<Eigen::Vector3d> &point_cloud) {
    if (point_cloud.empty())
      return;

    _root = std::make_shared<KDNode>(KDNode(std::make_shared<Eigen::Vector3d>(point_cloud[0]), 0));

    for (size_t i = 1; i < point_cloud.size(); i++) {
      auto new_node = std::make_shared<KDNode>(KDNode(std::make_shared<Eigen::Vector3d>(point_cloud[i]), i));
      auto curr_node = _root;
      bool done{false};
      size_t depth = 0;
      while (!done) {
        if (new_node->is_first_of(*curr_node, depth)) {
          if (curr_node->get_first() == nullptr) {
            new_node->set_parent(curr_node);
            curr_node->set_first(new_node);
            done = true;
            break;
          }
          curr_node = curr_node->get_first();
        } else {
          if (curr_node->get_second() == nullptr) {
            new_node->set_parent(curr_node);
            curr_node->set_second(new_node);
            done = true;
            break;
          }
          curr_node = curr_node->get_second();
        }
        depth++;
      }
    }
  }

private:
  std::shared_ptr<KDNode> _root;
};
