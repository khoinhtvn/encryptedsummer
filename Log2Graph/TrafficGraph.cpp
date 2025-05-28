//
// Created by lu on 5/7/25.
//

#include "includes/TrafficGraph.h"
#include <algorithm>

TrafficGraph::TrafficGraph() {}

TrafficGraph::~TrafficGraph() {}
std::shared_ptr<GraphNode> TrafficGraph::get_or_create_node(const std::string &id, const std::string &type) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        return it->second;
    }
    auto new_node = std::make_shared<GraphNode>(id, type);
    nodes_[id] = new_node;
    return new_node;
}

void TrafficGraph::add_node(std::shared_ptr<GraphNode> node) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    nodes_[node->id] = node;
}

void TrafficGraph::add_edge(std::shared_ptr<GraphEdge> edge) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    edges_.push_back(edge);
    // Update node degrees
    if (auto source_node = nodes_.find(edge->source); source_node != nodes_.end()) {
        source_node->second->features.degree++;
        source_node->second->features.out_degree++;
    }
    if (auto target_node = nodes_.find(edge->target); target_node != nodes_.end()) {
        target_node->second->features.degree++;
        target_node->second->features.in_degree++;
    }
}

std::shared_ptr<GraphNode> TrafficGraph::get_node(const std::string &id) const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    auto it = nodes_.find(id);
    if (it != nodes_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<GraphNode>> TrafficGraph::get_nodes() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    std::vector<std::shared_ptr<GraphNode>> node_list;
    for (const auto& pair : nodes_) {
        node_list.push_back(pair.second);
    }
    return node_list;
}

std::vector<std::shared_ptr<GraphEdge>> TrafficGraph::get_edges() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return edges_;
}

size_t TrafficGraph::get_node_count() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return nodes_.size();
}

size_t TrafficGraph::get_edge_count() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return edges_.size();
}

bool TrafficGraph::is_empty() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return nodes_.empty() && edges_.empty();
}

void TrafficGraph::aggregate_old_edges(std::chrono::seconds age_threshold) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    auto now = std::chrono::system_clock::now();

    auto it = edges_.begin();
    while (it != edges_.end()) {
        if (now - (*it)->last_seen > age_threshold) {
            // Aggregate data into the nodes
            if (auto source_node = nodes_.find((*it)->source); source_node != nodes_.end()) {
                if (auto target_node = nodes_.find((*it)->target); target_node != nodes_.end()) {
                    long long orig_bytes = 0;
                    long long resp_bytes = 0;
                    std::string protocol = "";
                    for (const auto& attr : (*it)->attributes) {
                        if (attr.first == "orig_bytes") {
                            try {
                                orig_bytes = std::stoll(attr.second);
                            } catch (...) {}
                        } else if (attr.first == "resp_bytes") {
                            try {
                                resp_bytes = std::stoll(attr.second);
                            } catch (...) {}
                        } else if (attr.first == "protocol") {
                            protocol = attr.second;
                        }
                    }
                    source_node->second->aggregate_historical_data(orig_bytes, resp_bytes, protocol);
                    target_node->second->aggregate_historical_data(resp_bytes, orig_bytes, protocol);
                }
            }
            it = edges_.erase(it);
        } else {
            ++it;
        }
    }
    recalculate_node_degrees();
}

void TrafficGraph::recalculate_node_degrees() {
    for (auto const& [node_id, node_ptr] : nodes_) {
        node_ptr->features.degree.store(0);
        node_ptr->features.in_degree.store(0);
        node_ptr->features.out_degree.store(0);
    }
    for (const auto& edge : edges_) {
        if (auto source_node = nodes_.find(edge->source); source_node != nodes_.end()) {
            source_node->second->features.degree++;
            source_node->second->features.out_degree++;
        }
        if (auto target_node = nodes_.find(edge->target); target_node != nodes_.end()) {
            target_node->second->features.degree++;
            target_node->second->features.in_degree++;
        }
    }
}