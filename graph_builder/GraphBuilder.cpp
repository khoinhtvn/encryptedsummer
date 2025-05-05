//
// Created by lu on 4/25/25.
//

#include "includes/GraphBuilder.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

void GraphNode::update_connection_features(const std::string &protocol, bool is_outgoing) {
    // Update degree counts
    ++features.degree;
    if (is_outgoing) {
        ++features.out_degree;
    } else {
        ++features.in_degree;
    }

    // Update protocol distribution
    features.protocol_counts[protocol]++;

    // Update temporal features
    auto now = std::chrono::system_clock::now();
    temporal.recent_connections.push_back(now);
    temporal.total_connections++; {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);
        temporal.minute_window.push(now);
        temporal.hour_window.push(now);
    }

    // Clean up old connections
    cleanup_time_windows();
    // Update activity score (weighted moving average)
    double new_activity = 1.0; // Base weight for new connection
    features.activity_score = 0.9 * features.activity_score.load() + 0.1 * new_activity;

    // Periodically clean old connections
    if (temporal.recent_connections.size() % 10 == 0) {
        cleanup_old_connections();
    }
}

void GraphNode::cleanup_old_connections() {
    auto now = std::chrono::system_clock::now();

    // Remove connections older than 1 hour
    auto hour_ago = now - std::chrono::hours(1);
    temporal.recent_connections.erase(
        std::remove_if(temporal.recent_connections.begin(),
                       temporal.recent_connections.end(),
                       [hour_ago](const auto &time) {
                           return time < hour_ago;
                       }),
        temporal.recent_connections.end()
    );
}

double GraphNode::calculate_anomaly_score() const {
    double last_min = get_connections_last_minute();
    double last_hour = get_connections_last_hour();
    double monitoring_minutes = std::chrono::duration<double>(
                                    std::chrono::system_clock::now() - temporal.monitoring_start).count() / 60.0;

    // Normalize hour count to per-minute rate
    double hour_rate = last_hour / std::min(60.0, monitoring_minutes);

    // Only calculate score if we have sufficient data
    if (monitoring_minutes < 1.0) return 0.0;

    // Calculate deviation from baseline
    if (hour_rate < 0.1) hour_rate = 0.1; // Minimum baseline
    double deviation = (last_min - hour_rate) / hour_rate;

    return std::min(1.0, std::max(0.0, 0.5 + deviation / 2.0));
}

void GraphNode::cleanup_time_windows() {
    auto now = std::chrono::system_clock::now();
    auto minute_ago = now - std::chrono::minutes(1);
    auto hour_ago = now - std::chrono::hours(1); {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);

        // Clean minute window
        while (!temporal.minute_window.empty() &&
               temporal.minute_window.front() < minute_ago) {
            temporal.minute_window.pop();
        }

        // Clean hour window
        while (!temporal.hour_window.empty() &&
               temporal.hour_window.front() < hour_ago) {
            temporal.hour_window.pop();
        }
    }
}

int GraphNode::get_connections_last_minute() const {
    std::lock_guard<std::mutex> lock(temporal.window_mutex);
    return temporal.minute_window.size();
}

int GraphNode::get_connections_last_hour() const {
    std::lock_guard<std::mutex> lock(temporal.window_mutex);
    return temporal.hour_window.size();
}


std::unique_ptr<GraphBuilder> GraphBuilder::instance = nullptr;
std::mutex GraphBuilder::instance_mutex;


void TrafficGraph::add_node(const std::string &id, const std::string &type) {
    std::unique_lock lock(graph_mutex);
    if (!nodes.contains(id)) {
        nodes[id] = std::make_shared<GraphNode>(id, type);
    }
}

GraphNode &TrafficGraph::get_or_create_node(const std::string &id, const std::string &type) {
    std::unique_lock lock(graph_mutex);

    if (!nodes.contains(id)) {
        const auto node = std::make_shared<GraphNode>(id, type);
        nodes[id] = node;
    }

    return *nodes[id];
}

void TrafficGraph::add_edge(const std::string &src, const std::string &tgt,
                            const std::string &rel, const std::unordered_map<std::string, std::string> &attrs) {
    //TODO: maybe in the future think about aggregating edges periodically. To reduce graph size and improve performance. Retain metadata such as connection_count, last_active, ports_used
    std::unique_lock lock(graph_mutex);
    auto edge = std::make_shared<GraphEdge>(src, tgt, rel);
    edge->attributes = attrs;
    edges.push_back(edge);
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
    return edges;
}

bool TrafficGraph::is_empty() const {
    return this->edges.empty();
}

TrafficGraph &GraphBuilder::get_graph() { return graph; }



void GraphBuilder::add_connection(const std::string &src_ip, const std::string &dst_ip,
                                 const std::string &proto, const std::string &timestamp,
                                 const int src_port, const int dst_port,
                                 const std::string &method,
                                 const std::string &host,
                                 const std::string &uri ,
                                 const std::string &version ,
                                 const std::string &user_agent ,
                                 const int request_body_len ,
                                 const int response_body_len,
                                 const int status_code ,
                                 const std::string &status_msg,
                                 const std::vector<std::string> &tags ,
                                 const std::vector<std::string> &resp_fuids ,
                                 const std::vector<std::string> &resp_mime_types ) {
    // Get or create nodes
    auto &src_node = graph.get_or_create_node(src_ip, "host");
    auto &dst_node = graph.get_or_create_node(dst_ip, "host");

    // Update node features
    src_node.update_connection_features(proto, true); // Outgoing connection
    dst_node.update_connection_features(proto, false); // Incoming connection

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
        for (const auto &tag : tags) {
            if (!tags_str.empty()) tags_str += ",";
            tags_str += tag;
        }
        attrs["tags"] = tags_str;
    }

    if (!resp_fuids.empty()) {
        std::string fuids_str;
        for (const auto &fuid : resp_fuids) {
            if (!fuids_str.empty()) fuids_str += ",";
            fuids_str += fuid;
        }
        attrs["resp_fuids"] = fuids_str;
    }

    if (!resp_mime_types.empty()) {
        std::string mime_str;
        for (const auto &mime : resp_mime_types) {
            if (!mime_str.empty()) mime_str += ",";
            mime_str += mime;
        }
        attrs["resp_mime_types"] = mime_str;
    }
    graph.add_edge(src_ip, dst_ip, proto + "_connection", attrs);
}

