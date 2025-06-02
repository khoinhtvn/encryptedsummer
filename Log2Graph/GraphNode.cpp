//
// Created by lu on 5/28/25.
//
#include "includes/NodeFeatureEncoder.h"
#include "includes/GraphNode.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "includes/EdgeFeatureEncoder.h"

GraphNode::TemporalFeatures::TemporalFeatures() :
    connections_last_minute(0),
    connections_last_hour(0),
    monitoring_start(std::chrono::system_clock::now()),
    total_connections(0)
    // window_mutex does not need explicit initialization as it's a mutable mutex,
    // its default constructor will initialize it.
    // minute_window and hour_window are queues, their default constructors will initialize them as empty.
    // first_seen and last_seen are strings, their default constructor will initialize them as empty.
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    first_seen = ss.str();
    last_seen = first_seen;
}

GraphNode::NodeFeatures::NodeFeatures() :
    degree(0),
in_degree(0),
out_degree(0),
activity_score(0.0),
total_connections_initiated(0),
total_connections_received(0),
ever_local_originated(false),
ever_local_responded(false),
total_orig_bytes(0),
total_resp_bytes(0),
total_orig_pkts(0),
total_resp_pkts(0),
avg_packet_size_sent(0.0),
avg_packet_size_received(0.0),
ever_ssl_curve_present(false),
ever_ssl_server_name_present(false),
ssl_resumption_count(0),
ever_ssl_last_alert_present(false),
ssl_established_count(0),
ever_ssl_history_present(false),
historical_total_orig_bytes(0),
historical_total_resp_bytes(0),
historical_total_connections(0)
{}
GraphNode::GraphNode(const std::string &id, const std::string &type)
    : id(id), type(type), last_connection_time(std::chrono::system_clock::now()) {
    temporal.monitoring_start = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(temporal.monitoring_start);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    temporal.first_seen = ss.str();
    temporal.last_seen = ss.str();
}

void GraphNode::update_connection_features(bool is_outgoing,
                                            const std::unordered_map<std::string, std::string>& connection_attributes) {
    std::lock_guard<std::mutex> lock(node_mutex);
    ++features.degree;
    if (is_outgoing) {
        ++features.out_degree;
        ++features.total_connections_initiated;
    } else {
        ++features.in_degree;
        ++features.total_connections_received;
    }

    if (connection_attributes.count("protocol")) {
        const std::string& protocol = connection_attributes.at("protocol");
        if (!protocol.empty()) {
            features.protocol_counts[protocol]++;
            features.protocols_used.insert(protocol);
        }
    }

    if (is_outgoing) {
        if (connection_attributes.count("id.resp_p")) {
            const std::string& remote_port = connection_attributes.at("id.resp_p");
            if (!remote_port.empty()) {
                features.remote_ports_connected_to.insert(remote_port);
            }
        }
        if (connection_attributes.count("id.orig_p")) {
            const std::string& local_port = connection_attributes.at("id.orig_p");
            if (!local_port.empty()) {
                features.local_ports_used.insert(local_port);
            }
        }
    } else {
        if (connection_attributes.count("id.orig_p")) {
            const std::string& remote_port = connection_attributes.at("id.orig_p");
            if (!remote_port.empty()) {
                features.remote_ports_connected_from.insert(remote_port);
            }
        }
        if (connection_attributes.count("id.resp_p")) {
            const std::string& local_port = connection_attributes.at("id.resp_p");
            if (!local_port.empty()) {
                features.local_ports_listening_on.insert(local_port);
            }
        }
    }

    if (connection_attributes.count("conn_state")) {
        const std::string& conn_state = connection_attributes.at("conn_state");
        if (!conn_state.empty()) {
            features.connection_state_counts[conn_state]++;
        }
    }

    if (connection_attributes.count("local_orig") && connection_attributes.at("local_orig") == "T") {
        features.ever_local_originated.store(true);
    }
    if (connection_attributes.count("local_resp") && connection_attributes.at("local_resp") == "T") {
        features.ever_local_responded.store(true);
    }

    long long current_orig_bytes = 0;
    if (connection_attributes.count("orig_bytes")) {
        const std::string& orig_bytes_str = connection_attributes.at("orig_bytes");
        if (!orig_bytes_str.empty()) {
            try {
                current_orig_bytes = std::stoll(orig_bytes_str);
                features.total_orig_bytes += current_orig_bytes;
            } catch (...) {}
        }
    }
    long long current_resp_bytes = 0;
    if (connection_attributes.count("resp_bytes")) {
        const std::string& resp_bytes_str = connection_attributes.at("resp_bytes");
        if (!resp_bytes_str.empty()) {
            try {
                current_resp_bytes = std::stoll(resp_bytes_str);
                features.total_resp_bytes += current_resp_bytes;
            } catch (...) {}
        }
    }
    if (connection_attributes.count("orig_pkts")) {
        const std::string& orig_pkts_str = connection_attributes.at("orig_pkts");
        if (!orig_pkts_str.empty()) {
            try {
                features.total_orig_pkts += std::stoll(orig_pkts_str);
            } catch (...) {}
        }
    }
    if (connection_attributes.count("resp_pkts")) {
        const std::string& resp_pkts_str = connection_attributes.at("resp_pkts");
        if (!resp_pkts_str.empty()) {
            try {
                features.total_resp_pkts += std::stoll(resp_pkts_str);
            } catch (...) {}
        }
    }

    if (features.total_orig_pkts > 0) {
        features.avg_packet_size_sent.store(static_cast<double>(features.total_orig_bytes) / features.total_orig_pkts);
    }
    if (features.total_resp_pkts > 0) {
        features.avg_packet_size_received.store(static_cast<double>(features.total_resp_bytes) / features.total_resp_pkts);
    }

    if (connection_attributes.count("service")) {
        const std::string& service_list = connection_attributes.at("service");
        if (!service_list.empty()) {
            std::stringstream ss(service_list);
            std::string service;
            while (std::getline(ss, service, ',')) {
                size_t first = service.find_first_not_of(" \t\n\r");
                if (std::string::npos == first) {
                    continue;
                }
                size_t last = service.find_last_not_of(" \t\n\r");
                features.services_used.insert(service.substr(first, (last - first + 1)));
            }
        }
    }

    if (connection_attributes.count("http_user_agent")) {
        const std::string& user_agent = connection_attributes.at("http_user_agent");
        if (!user_agent.empty()) {
            std::string categorized_user_agent = categorize_user_agent_string(user_agent);
            features.http_user_agent_counts[categorized_user_agent]++;
        }
    }
    if (connection_attributes.count("http_version")) {
        const std::string& http_version = connection_attributes.at("http_version");
        if (!http_version.empty()) {
            features.http_versions_used.insert(http_version);
            features.http_version_counts[http_version]++;
        }
    }
    if (connection_attributes.count("http_status_code")) {
        const std::string& status_code_str = connection_attributes.at("http_status_code");
        if (!status_code_str.empty()) {
            try {
                features.http_status_code_counts[std::stoi(status_code_str)]++;
            } catch (...) {}
        }
    }

    if (connection_attributes.count("ssl_version")) {
        const std::string& ssl_version = connection_attributes.at("ssl_version");
        if (!ssl_version.empty()) {
            features.ssl_versions_used.insert(ssl_version);
            features.ssl_version_counts[ssl_version]++;
        }
    }
    if (connection_attributes.count("ssl_cipher")) {
        const std::string& ssl_cipher = connection_attributes.at("ssl_cipher");
        if (!ssl_cipher.empty()) {
            features.ssl_ciphers_used.insert(ssl_cipher);
        }
    }
    if (connection_attributes.count("ssl_curve")) {
        const std::string& ssl_curve = connection_attributes.at("ssl_curve");
        if (!ssl_curve.empty()) {
            features.ever_ssl_curve_present.store(true);
        }
    }
    if (connection_attributes.count("ssl_server_name")) {
        const std::string& ssl_server_name = connection_attributes.at("ssl_server_name");
        if (!ssl_server_name.empty()) {
            features.ever_ssl_server_name_present.store(true);
        }
    }
    if (connection_attributes.count("ssl_resumed")) {
        const std::string& ssl_resumed = connection_attributes.at("ssl_resumed");
        if (!ssl_resumed.empty() && ssl_resumed == "true") {
            features.ssl_resumption_count++;
        }
    }
    if (connection_attributes.count("ssl_last_alert")) {
        const std::string& ssl_last_alert = connection_attributes.at("ssl_last_alert");
        if (!ssl_last_alert.empty()) {
            features.ever_ssl_last_alert_present.store(true);
        }
    }
    if (connection_attributes.count("ssl_next_protocol")) {
        const std::string& ssl_next_protocol = connection_attributes.at("ssl_next_protocol");
        if (!ssl_next_protocol.empty()) {
            features.ssl_next_protocols_used.insert(ssl_next_protocol);
        }
    }
    if (connection_attributes.count("ssl_established")) {
        const std::string& ssl_established = connection_attributes.at("ssl_established");
        if (!ssl_established.empty() && ssl_established == "true") {
            features.ssl_established_count++;
        }
    }
    if (connection_attributes.count("ssl_history")) {
        const std::string& ssl_history = connection_attributes.at("ssl_history");
        if (!ssl_history.empty()) {
            features.ever_ssl_history_present.store(true);
        }
    }

    auto now = std::chrono::system_clock::now();
    {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);
        temporal.minute_window.push(now);
        temporal.hour_window.push(now);
    }
    ++temporal.total_connections;
    temporal.connections_last_minute.store(temporal.minute_window.size());
    temporal.connections_last_hour.store(temporal.hour_window.size());

    double new_activity = 1.0;
    features.activity_score = 0.9 * features.activity_score.load() + 0.1 * new_activity;

    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    temporal.last_seen = ss.str();

    last_connection_time = now;
    connection_count++;
}

void GraphNode::cleanup_time_windows() {
    auto now = std::chrono::system_clock::now();
    auto minute_cutoff = now - std::chrono::seconds(60);
    auto hour_cutoff = now - std::chrono::hours(1);

    {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);
        while (!temporal.minute_window.empty() && temporal.minute_window.front() < minute_cutoff) {
            temporal.minute_window.pop();
        }
        while (!temporal.hour_window.empty() && temporal.hour_window.front() < hour_cutoff) {
            temporal.hour_window.pop();
        }
    }
    temporal.connections_last_minute.store(temporal.minute_window.size());
    temporal.connections_last_hour.store(temporal.hour_window.size());
}

int GraphNode::get_connections_last_minute() const {
    return temporal.connections_last_minute.load();
}

int GraphNode::get_connections_last_hour() const {
    return temporal.connections_last_hour.load();
}

double GraphNode::NodeFeatures::outgoing_connection_ratio() const {
    int total = total_connections_initiated + total_connections_received;
    if (total == 0) return 0.0;
    return static_cast<double>(total_connections_initiated) / total;
}

double GraphNode::NodeFeatures::incoming_connection_ratio() const {
    int total = total_connections_initiated + total_connections_received;
    if (total == 0) return 0.0;
    return static_cast<double>(total_connections_received) / total;
}

std::string GraphNode::NodeFeatures::most_frequent_protocol() const {
    std::string most_frequent = "N/A";
    int max_count = 0;
    if (protocol_counts.empty()) return most_frequent;
    for (const auto& pair : protocol_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_frequent = pair.first;
        }
    }
    return most_frequent;
}

size_t GraphNode::NodeFeatures::unique_remote_ports_connected_to() const {
    return remote_ports_connected_to.size();
}

size_t GraphNode::NodeFeatures::unique_local_ports_used() const {
    return local_ports_used.size();
}

std::string GraphNode::NodeFeatures::most_frequent_connection_state() const {
    std::string most_frequent = "N/A";
    int max_count = 0;
    if (connection_state_counts.empty()) return most_frequent;
    for (const auto& pair : connection_state_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_frequent = pair.first;
        }
    }
    return most_frequent;
}

bool GraphNode::NodeFeatures::connected_to_privileged_port() const {
    for (const auto& port_str : remote_ports_connected_to) {
        try {
            int port = std::stoi(port_str);
            if (port > 0 && port <= 1023) {
                return true;
            }
        } catch (...) {
            // Ignore parsing errors
        }
    }
    return false;
}

bool GraphNode::NodeFeatures::listened_on_privileged_port() const {
    for (const auto& port_str : local_ports_listening_on) {
        try {
            int port = std::stoi(port_str);
            if (port > 0 && port <= 1023) {
                return true;
            }
        } catch (...) {
            // Ignore parsing errors
        }
    }
    return false;
}

bool GraphNode::NodeFeatures::used_ssl_tls() const {
    return !ssl_versions_used.empty();
}

std::string GraphNode::escape_dot_string(const std::string &str) {
    std::string result = "";
    for (char c : str) {
        if (c == '"') {
            result += "\\\"";
        } else if (c == '\\') {
            result += "\\\\";
        } else {
            result += c;
        }
    }
    return result;
}

std::string GraphNode::to_dot_string() const {
    std::lock_guard<std::mutex> lock(node_mutex);
    std::stringstream ss;
    ss << "  \"" << escape_dot_string(id) << "\" [";
    ss << "degree=" << features.degree;
    ss << ", in_degree=" << features.in_degree;
    ss << ", out_degree=" << features.out_degree;
    ss << ", activity_score=" << std::fixed << std::setprecision(2) << features.activity_score.load();
    ss << ", total_connections=" << temporal.total_connections;

    ss << ", total_initiated=" << features.total_connections_initiated;
    ss << ", total_received=" << features.total_connections_received;
    ss << ", outgoing_ratio=" << std::fixed << std::setprecision(2) << features.outgoing_connection_ratio();
    ss << ", incoming_ratio=" << std::fixed << std::setprecision(2) << features.incoming_connection_ratio();
    ss << ", most_freq_proto=\"" << escape_dot_string(features.most_frequent_protocol()) << "\"";
    ss << ", unique_remote_ports_to=" << features.unique_remote_ports_connected_to();
    ss << ", unique_local_ports_used=" << features.unique_local_ports_used();
    ss << ", most_freq_conn_state=\"" << escape_dot_string(features.most_frequent_connection_state()) << "\"";
    ss << ", connected_to_priv_port=" << (features.connected_to_privileged_port() ? "true" : "false");
    ss << ", listened_on_priv_port=" << (features.listened_on_privileged_port() ? "true" : "false");
    ss << ", total_bytes_sent=" << features.total_bytes_sent();
    ss << ", total_bytes_received=" << features.total_bytes_received();
    ss << ", avg_pkt_size_sent=" << std::fixed << std::setprecision(2) << features.avg_packet_size_sent.load();
    ss << ", avg_pkt_size_received=" << std::fixed << std::setprecision(2) << features.avg_packet_size_received.load();
    ss << ", used_ssl_tls=" << (features.used_ssl_tls() ? "true" : "false");

    std::vector<std::pair<std::string, int>> sorted_protocols;
    for (const auto& pair : features.protocol_counts) {
        sorted_protocols.emplace_back(pair.first, pair.second);
    }
    std::sort(sorted_protocols.begin(), sorted_protocols.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    for (size_t i = 0; i < std::min((size_t)3, sorted_protocols.size()); ++i) {
        ss << ", top_proto_" << i + 1 << "=\"" << escape_dot_string(sorted_protocols[i].first) << "\"";
    }

    int service_count = 0;
    for (const auto& service : features.services_used) {
        ss << ", service_" << service_count + 1 << "=\"" << escape_dot_string(service) << "\"";
        if (++service_count >= 3) break;
    }

    std::vector<std::pair<std::string, int>> sorted_user_agents;
    for (const auto& pair : features.http_user_agent_counts) {
        sorted_user_agents.emplace_back(pair.first, pair.second);
    }
    std::sort(sorted_user_agents.begin(), sorted_user_agents.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    for (size_t i = 0; i < std::min((size_t)3, sorted_user_agents.size()); ++i) {
        ss << ", top_user_agent_" << i + 1 << "=\"" << escape_dot_string(sorted_user_agents[i].first) << "\"";
    }

    ss << ", first_seen=\"" << escape_dot_string(temporal.first_seen) << "\"";
    ss << ", last_seen=\"" << escape_dot_string(temporal.last_seen) << "\"";

    // Add historical features to the DOT string
    ss << ", historical_total_bytes_sent=" << features.historical_total_orig_bytes.load();
    ss << ", historical_total_bytes_received=" << features.historical_total_resp_bytes.load();
    ss << ", historical_total_connections=" << features.historical_total_connections.load();
    // Add top historical protocols (optional, can be more complex)
    std::vector<std::pair<std::string, int>> sorted_historical_protocols;
    for (const auto& pair : features.historical_protocol_counts) {
        sorted_historical_protocols.emplace_back(pair.first, pair.second);
    }
    std::sort(sorted_historical_protocols.begin(), sorted_historical_protocols.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    for (size_t i = 0; i < std::min((size_t)3, sorted_historical_protocols.size()); ++i) {
        ss << ", hist_proto_" << i + 1 << "=\"" << escape_dot_string(sorted_historical_protocols[i].first) << "\"";
    }

    ss << "];\n";
    return ss.str();
}

std::string GraphNode::to_dot_string_encoded() const {
    std::lock_guard<std::mutex> lock(node_mutex);
    std::stringstream ss;
    ss << "  \"" << escape_dot_string(id) << "\" [";

    std::vector<float> encoded_features_local = encoded_features;
    std::vector<std::string> feature_names = NodeFeatureEncoder::get_feature_names();

    if (!encoded_features_local.empty()) {
        size_t encoded_index = 0;
        for (const auto& feature_name : feature_names) {
           ss << ", " << feature_name << "=";
            if (feature_name == "most_freq_proto" || feature_name == "top_proto_1") {
                int hot_index = -1;
                for (size_t i = 0; i < NodeFeatureEncoder::PROTOCOLS.size(); ++i) {
                    if (encoded_features_local[encoded_index + i] == 1.0f) {
                        hot_index = static_cast<int>(i);
                        break;
                    }
                }
                ss << hot_index;
                encoded_index += NodeFeatureEncoder::PROTOCOLS.size();
            } else if (feature_name == "most_freq_conn_state") {
                int hot_index = -1;
                for (size_t i = 0; i < NodeFeatureEncoder::CONN_STATES.size(); ++i) {
                    if (encoded_features_local[encoded_index + i] == 1.0f) {
                        hot_index = static_cast<int>(i);
                        break;
                    }
                }
                ss << hot_index;
                encoded_index += NodeFeatureEncoder::CONN_STATES.size();
            } else if (feature_name.find("service") != std::string::npos) {
                int hot_index = -1;
                for (size_t i = 0; i < NodeFeatureEncoder::SERVICES.size(); ++i) {
                    if (encoded_features_local[encoded_index + i] == 1.0f) {
                        hot_index = static_cast<int>(i);
                        break;
                    }
                }
                ss << hot_index;
                encoded_index += NodeFeatureEncoder::SERVICES.size();
            } else if (feature_name == "top_user_agent_1") {
                int hot_index = -1;
                for (size_t i = 0; i < NodeFeatureEncoder::USER_AGENTS.size(); ++i) {
                    if (encoded_features_local[encoded_index + i] == 1.0f) {
                        hot_index = static_cast<int>(i);
                        break;
                    }
                }
                ss << hot_index;
                encoded_index += NodeFeatureEncoder::USER_AGENTS.size();
            } else if (feature_name == "most_freq_http_version") {
                int hot_index = -1;
                for (size_t i = 0; i < NodeFeatureEncoder::HTTP_VERSIONS.size(); ++i) {
                    if (encoded_features_local[encoded_index + i] == 1.0f) {
                        hot_index = static_cast<int>(i);
                        break;
                    }
                }
                ss << hot_index;
                encoded_index += NodeFeatureEncoder::HTTP_VERSIONS.size();
            } else if (feature_name == "ssl_version") {
                int hot_index = -1;
                for (size_t i = 0; i < NodeFeatureEncoder::SSL_VERSIONS.size(); ++i) {
                    if (encoded_features_local[encoded_index + i] == 1.0f) {
                        hot_index = static_cast<int>(i);
                        break;
                    }
                }
                ss << hot_index;
                encoded_index += NodeFeatureEncoder::SSL_VERSIONS.size();
            } else { // Handle scalar features
                ss << static_cast<int>(encoded_features_local[encoded_index]);
                encoded_index += 1;
            }
        }
    } else {
        ss << "encoded_features=\"[]\"";
    }

    ss << "];\n";
    return ss.str();
}


void GraphNode::aggregate_historical_data(long long orig_bytes, long long resp_bytes, const std::string& protocol) {
    features.historical_total_orig_bytes += orig_bytes;
    features.historical_total_resp_bytes += resp_bytes;
    features.historical_protocol_counts[protocol]++;
    features.historical_total_connections++;
}
void GraphNode::increment_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.degree++;
}

void GraphNode::decrement_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.degree--;
}

void GraphNode::increment_in_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.in_degree++;
}

void GraphNode::decrement_in_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.in_degree--;
}

void GraphNode::increment_out_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.out_degree++;
}

void GraphNode::decrement_out_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.out_degree--;
}

void GraphNode::reset_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.degree.store(0);
}

void GraphNode::reset_in_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.in_degree.store(0);
}

void GraphNode::reset_out_degree() {
    std::lock_guard<std::mutex> lock(node_mutex);
    features.out_degree.store(0);
}


void GraphNode::encode_features(const NodeFeatureEncoder& encoder) {
    encoded_features = encoder.encode_node_features(features, temporal);
}
