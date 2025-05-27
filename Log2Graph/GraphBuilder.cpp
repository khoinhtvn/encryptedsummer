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
                                    const int orig_ip_bytes, const int resp_ip_bytes,
                                    const std::string &version ,
                                    const std::string &cipher ,
                                    const std::string &curve ,
                                    const std::string &server_name ,
                                    const bool resumed ,
                                    const std::string &last_alert ,
                                    const std::string &next_protocol ,
                                    const bool established ,
                                    const std::string &ssl_history ,
                                    const std::string &cert_chain_fps ,
                                    const std::string &client_cert_chain_fps ,
                                    const bool sni_matches_cert ,
                                    const std::string &http_method ,
                                    const std::string &http_host ,
                                    const std::string &http_uri ,
                                    const std::string &http_referrer ,
                                    const std::string &http_version ,
                                    const std::string &http_user_agent ,
                                    const std::string &http_origin ,
                                    const std::string &http_status_code ,
                                    const std::string &http_username ) {
    // Get or create source and destination nodes in the graph.
    auto create_src = graph.get_or_create_node(src_ip, "");
    auto create_dst = graph.get_or_create_node(dst_ip, "");
    auto &src_node = create_src.first;
    auto &dst_node = create_dst.first;
    const bool src_created = create_src.second;
    const bool dst_created = create_dst.second;

    // Push update events for the source node to the update queue.
    // This informs other parts of the system about node creation or update.
    update_queue.push({
        src_created ? GraphUpdate::Type::NODE_CREATE : GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(src_ip),
        std::weak_ptr<GraphEdge>()
    });

    // Push update events for the destination node to the update queue.
    // Similar to the source node, this notifies about its creation or update.
    update_queue.push({
        dst_created ? GraphUpdate::Type::NODE_CREATE : GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(dst_ip),
        std::weak_ptr<GraphEdge>()
    });

    // Update connection-related features of the source and destination nodes.
    // This might involve tracking the protocols they've communicated with.
    src_node.update_connection_features(proto, true); // Outgoing connection
    dst_node.update_connection_features(proto, false); // Incoming connection

    // Create a map of attributes to store connection details.
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
        {"resp_ip_bytes", std::to_string(resp_ip_bytes)},
        {"ssl_version", version},
        {"ssl_cipher", cipher},
        {"ssl_curve", curve},
        {"ssl_server_name", server_name},
        {"ssl_resumed", resumed ? "true" : "false"},
        {"ssl_last_alert", last_alert},
        {"ssl_next_protocol", next_protocol},
        {"ssl_established", established ? "true" : "false"},
        {"ssl_history", ssl_history},
        {"ssl_cert_chain_fps", cert_chain_fps},
        {"ssl_client_cert_chain_fps", client_cert_chain_fps},
        {"ssl_sni_matches_cert", sni_matches_cert ? "true" : "false"},
        {"http_method", http_method},
        {"http_host", http_host},
        {"http_uri", http_uri},
        {"http_referrer", http_referrer},
        {"http_version", http_version},
        {"http_user_agent", http_user_agent},
        {"http_origin", http_origin},
        {"http_status_code", http_status_code},
        {"http_username", http_username}
    };

    // Encode the connection attributes into a feature vector.
    std::vector<float> features = feature_encoder.encode_features(attrs);

    // Add an edge representing the connection to the graph.
    auto edge = graph.add_edge(src_ip, dst_ip, proto, attrs, features);

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
