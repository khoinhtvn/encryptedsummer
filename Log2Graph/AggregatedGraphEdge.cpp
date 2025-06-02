//
// Created by lu on 6/2/25.
//

#include "includes/AggregatedGraphEdge.h"
#include <sstream>
#include <iomanip>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min

AggregatedGraphEdge::AggregatedGraphEdge(const std::string &src, const std::string &tgt,
                                         const std::string &proto, const std::string &service, const std::string &dport)
    : source(src), target(tgt), protocol(proto), service(service), dst_port(dport),
      total_orig_bytes(0), total_resp_bytes(0),
      total_orig_pkts(0), total_resp_pkts(0),
      total_orig_ip_bytes(0), total_resp_ip_bytes(0),
      connection_count(1),
      first_seen(std::chrono::system_clock::now()), last_seen(std::chrono::system_clock::now()) {
}

void AggregatedGraphEdge::update(const std::unordered_map<std::string, std::string> &raw_feature_map,
                                const std::vector<float> &encoded_features) {
    connection_count++;
    auto now = std::chrono::system_clock::now();
    if (now < first_seen) { // Update first_seen if a connection with an earlier timestamp arrives
        first_seen = now;
    }
    last_seen = now;

    // --- Aggregate Numerical Features (Sums) ---
    auto safe_stoll = [](const std::string& s) {
        try { return std::stoll(s); } catch (...) { return 0LL; }
    };

    if (raw_feature_map.count("orig_bytes")) total_orig_bytes += safe_stoll(raw_feature_map.at("orig_bytes"));
    if (raw_feature_map.count("resp_bytes")) total_resp_bytes += safe_stoll(raw_feature_map.at("resp_bytes"));
    if (raw_feature_map.count("orig_pkts")) total_orig_pkts += safe_stoll(raw_feature_map.at("orig_pkts"));
    if (raw_feature_map.count("resp_pkts")) total_resp_pkts += safe_stoll(raw_feature_map.at("resp_pkts"));
    if (raw_feature_map.count("orig_ip_bytes")) total_orig_ip_bytes += safe_stoll(raw_feature_map.at("orig_ip_bytes"));
    if (raw_feature_map.count("resp_ip_bytes")) total_resp_ip_bytes += safe_stoll(raw_feature_map.at("resp_ip_bytes"));


    // --- Aggregate Encoded Features (Sum) ---
    if (aggregated_encoded_features.empty()) {
        aggregated_encoded_features = encoded_features;
    } else if (!encoded_features.empty() && aggregated_encoded_features.size() == encoded_features.size()) {
        for (size_t i = 0; i < aggregated_encoded_features.size(); ++i) {
            aggregated_encoded_features[i] += encoded_features[i];
        }
    }
}

std::string AggregatedGraphEdge::escape_dot_string(const std::string &str) {
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

std::string AggregatedGraphEdge::to_dot_string_aggregated() const {
    std::stringstream ss;
    ss << "  \"" << escape_dot_string(source) << "\" -> \"" << escape_dot_string(target) << "\" [";
    ss << "protocol=\"" << escape_dot_string(protocol) << "\"";
    ss << ", service=\"" << escape_dot_string(service) << "\"";
    ss << ", dst_port=\"" << escape_dot_string(dst_port) << "\"";
    ss << ", count=" << connection_count;

    // Numerical Sums
    ss << ", total_orig_bytes=" << total_orig_bytes;
    ss << ", total_resp_bytes=" << total_resp_bytes;
    ss << ", total_orig_pkts=" << total_orig_pkts;
    ss << ", total_resp_pkts=" << total_resp_pkts;
    ss << ", total_orig_ip_bytes=" << total_orig_ip_bytes;
    ss << ", total_resp_ip_bytes=" << total_resp_ip_bytes;
/*
    // Aggregated Encoded Features (sum - first 8)
    if (!aggregated_encoded_features.empty()) {
        ss << ", encoded_features_sum=\"";
        for (size_t i = 0; i < std::min(aggregated_encoded_features.size(), (size_t)8); ++i) {
            ss << std::fixed << std::setprecision(4) << aggregated_encoded_features[i]; // More precision for floats
            if (i < std::min(aggregated_encoded_features.size(), (size_t)8) - 1) ss << ",";
        }
        if (aggregated_encoded_features.size() > 8) {
            ss << ",...";
        }
        ss << "\"";
    }

    auto first_seen_t = std::chrono::system_clock::to_time_t(first_seen);
    auto last_seen_t = std::chrono::system_clock::to_time_t(last_seen);
    std::stringstream ss_first, ss_last;
    ss_first << std::put_time(std::localtime(&first_seen_t), "%Y-%m-%d %H:%M:%S");
    ss_last << std::put_time(std::localtime(&last_seen_t), "%Y-%m-%d %H:%M:%S");
    ss << ", first_seen=\"" << ss_first.str() << "\"";
    ss << ", last_seen=\"" << ss_last.str() << "\"";
    */
    ss << "];\n";

    return ss.str();
}

std::string AggregatedGraphEdge::to_dot_string() const {
    return to_dot_string_aggregated();
}