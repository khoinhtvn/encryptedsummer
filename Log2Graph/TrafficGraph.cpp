#include "includes/TrafficGraph.h"
#include "includes/AggregatedGraphEdge.h" // Include for AggregatedGraphEdge
#include <algorithm>
#include <iostream>
#include <mutex> // Include mutex here as well

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

void TrafficGraph::add_aggregated_edge(const AggregatedGraphEdge& new_edge) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    std::string src_id = new_edge.source;
    std::string dest_id = new_edge.target;

    auto source_it = nodes_.find(src_id);
    if (source_it == nodes_.end()) {
        std::cerr << "Error: Source node with ID '" << src_id << "' not found for aggregated edge." << std::endl;
        return;
    }
    auto dest_it = nodes_.find(dest_id);
    if (dest_it == nodes_.end()) {
        std::cerr << "Error: Destination node with ID '" << dest_id << "' not found for aggregated edge." << std::endl;
        return;
    }

    bool found = false;
    for (auto& existing_edge : aggregated_edges_) {
        if (existing_edge.source == new_edge.source &&
            existing_edge.target == new_edge.target &&
            existing_edge.protocol == new_edge.protocol &&
            existing_edge.service == new_edge.service &&
            existing_edge.dst_port == new_edge.dst_port) {
            // Update the existing edge
            existing_edge.connection_count += new_edge.connection_count;
            existing_edge.total_orig_bytes += new_edge.total_orig_bytes;
            existing_edge.total_resp_bytes += new_edge.total_resp_bytes;
            existing_edge.total_orig_pkts += new_edge.total_orig_pkts;
            existing_edge.total_resp_pkts += new_edge.total_resp_pkts;
            existing_edge.total_orig_ip_bytes += new_edge.total_orig_ip_bytes;
            existing_edge.total_resp_ip_bytes += new_edge.total_resp_ip_bytes;
            existing_edge.last_seen = new_edge.last_seen;
            // Potentially update aggregated_encoded_features as well
            found = true;
            break;
        }
    }

    if (!found) {
        aggregated_edges_.push_back(new_edge);
        source_it->second->increment_out_degree();
        source_it->second->increment_degree();
        dest_it->second->increment_in_degree();
        dest_it->second->increment_degree();
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

std::vector<AggregatedGraphEdge> TrafficGraph::get_aggregated_edges() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return aggregated_edges_;
}

size_t TrafficGraph::get_node_count() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return nodes_.size();
}

size_t TrafficGraph::get_aggregated_edge_count() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return aggregated_edges_.size();
}

bool TrafficGraph::is_empty() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return nodes_.empty() && aggregated_edges_.empty();
}

void TrafficGraph::aggregate_old_edges(std::chrono::seconds age_threshold) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    // This method is likely no longer needed as edge aggregation happens in GraphBuilder.
    // You might want to remove this method or repurpose it for other cleanup tasks.
    std::cerr << "Warning: TrafficGraph::aggregate_old_edges called, but edge aggregation is now handled in GraphBuilder." << std::endl;
}

void TrafficGraph::recalculate_node_degrees() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    // Reset all node degrees
    for (auto const& [node_id, node_ptr] : nodes_) {
        node_ptr->reset_degree();
        node_ptr->reset_in_degree();
        node_ptr->reset_out_degree();
    }

    // Recalculate degrees based on aggregated edges
    for (const auto& edge : aggregated_edges_) {
        std::string source_id = edge.source;
        std::string target_id = edge.target;

        if (auto source_node = nodes_.find(source_id); source_node != nodes_.end()) {
            source_node->second->increment_out_degree();
            source_node->second->increment_degree();
        } else {
            std::cerr << "Warning: Source node '" << source_id << "' not found during degree recalculation." << std::endl;
        }

        if (auto target_node = nodes_.find(target_id); target_node != nodes_.end()) {
            target_node->second->increment_in_degree();
            target_node->second->increment_degree();
        } else {
            std::cerr << "Warning: Target node '" << target_id << "' not found during degree recalculation." << std::endl;
        }
    }
}