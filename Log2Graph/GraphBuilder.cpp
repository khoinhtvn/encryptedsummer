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

void GraphBuilder::add_connection(const std::string &src_ip, const std::string &dst_ip,
                                  const std::string &proto, const std::string &service, const std::string &timestamp,
                                  const int src_port, const int dst_port,
                                  const int orig_bytes, const int resp_bytes,
                                  const std::string &conn_state,
                                  const bool local_orig, const bool local_resp,
                                  const std::string &history,
                                  const int orig_pkts, const int resp_pkts,
                                  const int orig_ip_bytes, const int resp_ip_bytes) {
    // Get or create nodes
    auto create_src = graph.get_or_create_node(src_ip, "");
    auto create_dst = graph.get_or_create_node(dst_ip, "");
    auto &src_node = create_src.first;
    auto &dst_node = create_dst.first;
    const bool src_created = create_src.second;
    const bool dst_created = create_dst.second;

    update_queue.push({
        src_created ? GraphUpdate::Type::NODE_CREATE : GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(src_ip),
        std::weak_ptr<GraphEdge>()
    });

    update_queue.push({
        dst_created ? GraphUpdate::Type::NODE_CREATE : GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(dst_ip),
        std::weak_ptr<GraphEdge>()
    });

    // Update node features
    src_node.update_connection_features(proto, true); // Outgoing connection
    dst_node.update_connection_features(proto, false); // Incoming connection

    // Add edge with updated attributes
    std::unordered_map<std::string, std::string> attrs = {
        {"protocol", proto},
        {"timestamp", timestamp},
        {"src_port", std::to_string(src_port)},
        {"dst_port", std::to_string(dst_port)},
            {"service", service},
        {"conn_state", conn_state},
        {"local_orig", local_orig ? "true" : "false"},
        {"local_resp", local_resp ? "true" : "false"},
        {"history", history},
        {"orig_bytes", std::to_string(orig_bytes)},
        {"resp_bytes", std::to_string(resp_bytes)},
        {"orig_pkts", std::to_string(orig_pkts)},
        {"resp_pkts", std::to_string(resp_pkts)},
        {"orig_ip_bytes", std::to_string(orig_ip_bytes)},
        {"resp_ip_bytes", std::to_string(resp_ip_bytes)}
    };

    std::vector<float> features = feature_encoder.encode_features(attrs);

    auto edge = graph.add_edge(src_ip, dst_ip, proto, attrs, features);
    update_queue.push({
        GraphUpdate::Type::EDGE_CREATE,
        std::weak_ptr<GraphNode>(),
        edge
    });
}

TrafficGraph &GraphBuilder::get_graph() {
    return graph;
}
