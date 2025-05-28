// ... other includes ...
#include <memory>
#include <mutex>
#include <chrono>

#include "GraphEdge.h"
#include "GraphNode.h"

#ifndef TRAFFICGRAPH_H
#define TRAFFICGRAPH_H


class TrafficGraph {
public:
    TrafficGraph();
    ~TrafficGraph();

    std::shared_ptr<GraphNode> get_or_create_node(const std::string &id, const std::string &type);
    void add_node(std::shared_ptr<GraphNode> node);
    void add_edge(std::shared_ptr<GraphEdge> edge);
    std::shared_ptr<GraphNode> get_node(const std::string &id) const;
    std::vector<std::shared_ptr<GraphNode>> get_nodes() const;
    std::vector<std::shared_ptr<GraphEdge>> get_edges() const;
    size_t get_node_count() const;
    size_t get_edge_count() const;
    bool is_empty() const;
    void aggregate_old_edges(std::chrono::seconds age_threshold);

private:
    std::unordered_map<std::string, std::shared_ptr<GraphNode>> nodes_;
    std::vector<std::shared_ptr<GraphEdge>> edges_;
    mutable std::mutex graph_mutex_; // Make mutex mutable
    void recalculate_node_degrees();
};
#endif // TRAFFICGRAPH_H