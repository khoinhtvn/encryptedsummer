//
// Created by lu on 4/25/25.
//

#include "includes/GraphBuilder.h"
#include "includes/AggregatedGraphEdge.h" // Include AggregatedGraphEdge
#include "includes/GraphNode.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <tuple>

std::unique_ptr<GraphBuilder> GraphBuilder::instance = nullptr;
std::mutex GraphBuilder::instance_mutex;

TrafficGraph &GraphBuilder::get_graph() {
    return graph;
}

void GraphBuilder::add_connection(const std::unordered_map<std::string, std::string> &raw_feature_map,
                                  std::vector<float> &encoded_features) {
    if (!raw_feature_map.count("src_ip") || !raw_feature_map.count("dst_ip") ||
        !raw_feature_map.count("protocol") || !raw_feature_map.count("service") ||
        !raw_feature_map.count("dst_port")) {
        std::cerr << "Error: Missing essential features (src_ip, dst_ip, protocol, service, dst_port) in connection data." << std::endl;
        return;
    }

    const std::string& src_ip = raw_feature_map.at("src_ip");
    const std::string& dst_ip = raw_feature_map.at("dst_ip");
    const std::string& protocol = raw_feature_map.at("protocol");
    std::string service_str = raw_feature_map.at("service"); // Contains "quic,ssl"
    const std::string& dst_port = raw_feature_map.at("dst_port");

    auto src_node_ptr = graph.get_or_create_node(src_ip, "host");
    auto dst_node_ptr = graph.get_or_create_node(dst_ip, "host");

    std::stringstream ss(service_str);
    std::string individual_service;

    while (std::getline(ss, individual_service, ',')) {
        // Trim leading and trailing whitespace from the service
        individual_service.erase(0, individual_service.find_first_not_of(" "));
        individual_service.erase(individual_service.find_last_not_of(" ") + 1);

        auto key = std::make_tuple(src_ip, dst_ip, protocol, individual_service, dst_port);

        if (aggregated_edges.count(key)) {
            aggregated_edges.at(key).update(raw_feature_map, encoded_features);
        } else {
            aggregated_edges.emplace(key, AggregatedGraphEdge(src_ip, dst_ip, protocol, individual_service, dst_port));
            aggregated_edges.at(key).update(raw_feature_map, encoded_features);
        }

        graph.add_aggregated_edge(aggregated_edges.at(key));
    }

    src_node_ptr->update_connection_features(true, raw_feature_map);
    dst_node_ptr->update_connection_features(false, raw_feature_map);

    src_node_ptr->encode_features(node_encoder);
    dst_node_ptr->encode_features(node_encoder);
}