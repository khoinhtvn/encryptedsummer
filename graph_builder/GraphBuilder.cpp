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
                                  const std::string &proto, const std::string &timestamp,
                                  const int src_port, const int dst_port,
                                  const std::string &method,
                                  const std::string &host,
                                  const std::string &uri,
                                  const std::string &version,
                                  const std::string &user_agent,
                                  const int request_body_len,
                                  const int response_body_len,
                                  const int status_code,
                                  const std::string &status_msg,
                                  const std::vector<std::string> &tags,
                                  const std::vector<std::string> &resp_fuids,
                                  const std::vector<std::string> &resp_mime_types) {
    // Get or create nodes
    auto &src_node = graph.get_or_create_node(src_ip, host);
    auto &dst_node = graph.get_or_create_node(dst_ip, host);

    // Update node features
    src_node.update_connection_features(proto, true); // Outgoing connection
    dst_node.update_connection_features(proto, false); // Incoming connection

    update_queue.push({
        GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(src_ip),
        std::weak_ptr<GraphEdge>()
    });

    update_queue.push({
        GraphUpdate::Type::NODE_UPDATE,
        graph.get_node_reference(dst_ip),
        std::weak_ptr<GraphEdge>()
    });

    // Add edge
    std::unordered_map<std::string, std::string> attrs = {
        {"protocol", proto},
        {"timestamp", timestamp},
        {"src_port", std::to_string(src_port)},
        {"dst_port", std::to_string(dst_port)},
        {"method", method},
        {"host", host},
        {"uri", uri},
        {"version", version},
        {"user_agent", user_agent},
        {"request_body_len", std::to_string(request_body_len)},
        {"response_body_len", std::to_string(response_body_len)},
        {"status_code", std::to_string(status_code)},
        {"status_msg", status_msg}
    };
    // Handle vector attributes by joining them with commas
    if (!tags.empty()) {
        std::string tags_str;
        for (const auto &tag: tags) {
            if (!tags_str.empty()) tags_str += ",";
            tags_str += tag;
        }
        attrs["tags"] = tags_str;
    }

    if (!resp_fuids.empty()) {
        std::string fuids_str;
        for (const auto &fuid: resp_fuids) {
            if (!fuids_str.empty()) fuids_str += ",";
            fuids_str += fuid;
        }
        attrs["resp_fuids"] = fuids_str;
    }

    if (!resp_mime_types.empty()) {
        std::string mime_str;
        for (const auto &mime: resp_mime_types) {
            if (!mime_str.empty()) mime_str += ",";
            mime_str += mime;
        }
        attrs["resp_mime_types"] = mime_str;
    }
    auto edge = graph.add_edge(src_ip, dst_ip, proto + "_connection", attrs);
    update_queue.push({
        GraphUpdate::Type::EDGE_UPDATE,
        std::weak_ptr<GraphNode>(),
        edge
    });
}


