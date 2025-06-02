//
// Created by lu on 4/25/25.
//

#include "includes/GraphBuilder.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>


std::unique_ptr<GraphBuilder> GraphBuilder::instance = nullptr;
std::mutex GraphBuilder::instance_mutex;
void GraphBuilder::add_connection(const std::unordered_map<std::string, std::string> &raw_feature_map,
                                   std::vector<float> &encoded_features) {
    if (!raw_feature_map.count("src_ip") || !raw_feature_map.count("dst_ip")) {
        std::cerr << "Error: Missing source or destination IP in connection data." << std::endl;
        return;
    }

    const std::string& src_ip = raw_feature_map.at("src_ip");
    const std::string& dst_ip = raw_feature_map.at("dst_ip");

    auto src_node_ptr = graph.get_or_create_node(src_ip, "host");
    auto dst_node_ptr = graph.get_or_create_node(dst_ip, "host");

    update_queue.push({
        GraphUpdate::Type::NODE_CREATE ,
        src_node_ptr,
        std::weak_ptr<GraphEdge>()
    });

    update_queue.push({
        GraphUpdate::Type::NODE_CREATE,
        dst_node_ptr,
        std::weak_ptr<GraphEdge>()
    });

    std::string relationship = "connects_to";
    if (raw_feature_map.count("proto")) {
        relationship = raw_feature_map.at("proto") + "_" + relationship;
    }

    auto new_edge = std::make_shared<GraphEdge>(src_ip, dst_ip, relationship);
    new_edge->encoded_features = encoded_features;
    new_edge->attributes = raw_feature_map;
    graph.add_edge(new_edge);

    update_queue.push({
        GraphUpdate::Type::EDGE_CREATE,
        std::weak_ptr<GraphNode>(),
        new_edge
    });

    src_node_ptr->update_connection_features(true, raw_feature_map);
    dst_node_ptr->update_connection_features(false, raw_feature_map);

    // Encode node features and store them in the GraphNode object
    src_node_ptr->encode_features(node_encoder);
    dst_node_ptr->encode_features(node_encoder);

}

TrafficGraph &GraphBuilder::get_graph() {
    return graph;
}
