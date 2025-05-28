//
// Created by lu on 5/7/25.
//
#include "includes/TrafficGraph.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "includes/GraphBuilder.h"
#include "includes/GraphEdge.h"
#include "includes/GraphNode.h"

void TrafficGraph::add_node(const std::string &id, const std::string &type) {
    std::unique_lock lock(graph_mutex);
    if (!nodes.contains(id)) {
        nodes[id] = std::make_shared<GraphNode>(id, type);
    }
}

std::pair<GraphNode &, bool> TrafficGraph::get_or_create_node(const std::string &id, const std::string &type) {
    std::unique_lock lock(graph_mutex);
    bool create = !nodes.contains(id);
    if (create) {
        const auto node = std::make_shared<GraphNode>(id, type);
        nodes[id] = node;
        ++update_counter;
    }

    return {*nodes[id], create};
}

std::weak_ptr<GraphNode> TrafficGraph::get_node_reference(const std::string &id) {
    std::unique_lock lock(graph_mutex);

    if (!nodes.contains(id)) {
        throw std::runtime_error("Node not found");
    }

    return std::weak_ptr(nodes[id]);
}

std::weak_ptr<GraphEdge> TrafficGraph::add_edge(const std::string &src, const std::string &tgt,
                                                const std::string &rel,
                                                const std::unordered_map<std::string, std::string> &attrs, const std::vector<float> &encoded_features) {
    //TODO: maybe in the future think about aggregating edges periodically. To reduce graph size and improve performance. Retain metadata such as connection_count, last_active, ports_used
    std::unique_lock lock(graph_mutex);
    auto edge = std::make_shared<GraphEdge>(src, tgt, rel);
    edge->attributes = attrs;
    edge->encoded_features = std::move(encoded_features);
    edges.push_back(edge);
    return std::weak_ptr(edge);
}

// Thread-safe graph access methods
std::vector<std::shared_ptr<GraphNode> > TrafficGraph::get_nodes() const {
    std::shared_lock lock(graph_mutex);
    std::vector<std::shared_ptr<GraphNode> > result;
    for (const auto &pair: nodes) {
        result.push_back(pair.second);
    }
    return result;
}

std::vector<std::shared_ptr<GraphEdge> > TrafficGraph::get_edges() const {
    std::shared_lock lock(graph_mutex);
    return edges;
}

bool TrafficGraph::is_empty() const {
    std::shared_lock lock(graph_mutex);
    return this->edges.empty();
}

