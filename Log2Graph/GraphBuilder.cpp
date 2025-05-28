//
// Created by lu on 4/25/25.
//

#include "includes/GraphBuilder.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>


std::unique_ptr<GraphBuilder> GraphBuilder::instance = nullptr;
std::mutex GraphBuilder::instance_mutex;

void GraphBuilder::add_connection(const std::unordered_map<std::string, std::string> &raw_feature_map,
                                  std::vector<float> &encoded_features) {
    const std::string& src_ip = raw_feature_map.at("src_ip");
    const std::string& dst_ip = raw_feature_map.at("dst_ip");
    const std::string& proto = raw_feature_map.at("protocol");

    // Get or create source and destination nodes in the graph.
    auto create_src = graph.get_or_create_node(src_ip, "");
    auto create_dst = graph.get_or_create_node(dst_ip, "");
    auto &src_node = create_src.first;
    auto &dst_node = create_dst.first;
    const bool src_created = create_src.second;
    const bool dst_created = create_dst.second;

    // Push update events for the source node to the update queue.
    update_queue.push({
        src_created ? GraphUpdate::Type::NODE_CREATE : GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(src_ip),
        std::weak_ptr<GraphEdge>()
    });

    // Push update events for the destination node to the update queue.
    update_queue.push({
        dst_created ? GraphUpdate::Type::NODE_CREATE : GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(dst_ip),
        std::weak_ptr<GraphEdge>()
    });

    // Update connection-related features of the source and destination nodes.
    // You might want to extract relevant information from the raw_feature_map
    // to update node-level features based on the connection.
    if (raw_feature_map.count("protocol")) {
        src_node.update_connection_features(raw_feature_map.at("protocol"), true); // Outgoing
        dst_node.update_connection_features(raw_feature_map.at("protocol"), false); // Incoming
    }

    // Add an edge representing the connection to the graph.
    auto edge = graph.add_edge(src_ip, dst_ip, proto, raw_feature_map, encoded_features);

    // Push an update event for the newly created edge to the update queue.
    update_queue.push({
        GraphUpdate::Type::EDGE_CREATE,
        std::weak_ptr<GraphNode>(),
        edge
    });
}

TrafficGraph &GraphBuilder::get_graph() {
    return graph;
}
