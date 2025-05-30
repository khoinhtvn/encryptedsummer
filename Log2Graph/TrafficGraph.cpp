//
// Created by lu on 5/7/25.
//

#include "includes/TrafficGraph.h"
#include <algorithm>
#include <iostream>

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
    edges_.push_back(edge);
    std::shared_ptr<GraphNode> source_node = nodes_[edge->get_source_node_id()];
    std::shared_ptr<GraphNode> dest_node = nodes_[edge->get_destination_node_id()];

    source_node->increment_out_degree();
    source_node->increment_degree();
    dest_node->increment_in_degree();
    dest_node->increment_degree();
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
    // Reset all node degrees to 0 using public methods
    for (auto const& [node_id, node_ptr] : nodes_) {
        node_ptr->reset_degree();
        node_ptr->reset_in_degree();
        node_ptr->reset_out_degree();
    }

    // Recalculate degrees by iterating through all edges
    for (const auto& edge : edges_) {
        std::string source_id = edge->get_source_node_id();
        std::string target_id = edge->get_destination_node_id();

        // Retrieve nodes using get_node to ensure existence and proper shared_ptr handling
        std::shared_ptr<GraphNode> source_node = get_node(source_id);
        std::shared_ptr<GraphNode> target_node = get_node(target_id);

        if (source_node) {
            source_node->increment_out_degree();
            source_node->increment_degree();
        } else {
            std::cerr << "Warning: Source node '" << source_id << "' not found during recalculation." << std::endl;
        }

        if (target_node) {
            target_node->increment_in_degree();
            target_node->increment_degree();
        } else {
            std::cerr << "Warning: Target node '" << target_id << "' not found during recalculation." << std::endl;
        }
    }
}