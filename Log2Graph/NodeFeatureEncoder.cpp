#include "includes/NodeFeatureEncoder.h"
#include <algorithm>
#include <cmath> // For std::log and potential normalization

// Define fixed vocabularies (same as before)
const std::vector<std::string> PROTOCOLS = {"tcp", "udp", "icmp"};
const std::vector<std::string> CONN_STATES = {"S0", "S1", "SF", "REJ", "RSTO", "RSTR", "OTH", "SH", "SHR", "RSTOS0", "RSTRH"};
const std::vector<std::string> SERVICES = {"-", "http", "ssl", "dns", "ftp", "ssh", "rdp", "smb"};
const std::vector<std::string> USER_AGENTS = {"Mozilla", "Chrome", "Safari", "Edge", "curl", "Wget", "Python", "Java", "Unknown"};
const std::vector<std::string> BOOLEANS = {"false", "true"};

NodeFeatureEncoder::NodeFeatureEncoder() {
    // Initialize protocol vocabulary
    for (size_t i = 0; i < PROTOCOLS.size(); ++i) {
        protocol_vocab_[PROTOCOLS[i]] = i;
    }
    protocol_vec_size_ = PROTOCOLS.size() + 1; // +1 for 'other'

    // Initialize connection state vocabulary
    for (size_t i = 0; i < CONN_STATES.size(); ++i) {
        conn_state_vocab_[CONN_STATES[i]] = i;
    }
    conn_state_vec_size_ = CONN_STATES.size() + 1; // +1 for 'other'

    // Initialize service vocabulary
    for (size_t i = 0; i < SERVICES.size(); ++i) {
        service_vocab_[SERVICES[i]] = i;
    }
    service_vec_size_ = SERVICES.size() + 1; // +1 for 'other'

    // Initialize user agent vocabulary (including "Unknown")
    for (size_t i = 0; i < USER_AGENTS.size(); ++i) {
        user_agent_vocab_[USER_AGENTS[i]] = i;
    }
    user_agent_vec_size_ = USER_AGENTS.size(); // No 'other' needed as we explicitly include "Unknown"

    // Initialize boolean vocabulary
    for (size_t i = 0; i < BOOLEANS.size(); ++i) {
        bool_vocab_[BOOLEANS[i]] = i;
    }
    bool_vec_size_ = BOOLEANS.size();
}

std::vector<float> NodeFeatureEncoder::one_hot_encode(const std::string& value,
                                                  const std::map<std::string, int>& vocabulary,
                                                  size_t vector_size) const {
    std::vector<float> encoded_vector(vector_size, 0.0f);
    auto it = vocabulary.find(value);
    if (it != vocabulary.end()) {
        encoded_vector[it->second] = 1.0f;
    } else {
        // If value not found, map to the last category (which is implicitly 'other' for categorical, or handled by default for booleans)
        encoded_vector[vector_size - 1] = 1.0f;
    }
    return encoded_vector;
}

std::vector<float> NodeFeatureEncoder::encode_node_features(const struct GraphNode::NodeFeatures& features) const {
    std::vector<float> encoded_features;

    // 1. most_freq_proto
    encoded_features.insert(encoded_features.end(),
                            one_hot_encode(features.most_frequent_protocol(), protocol_vocab_, protocol_vec_size_).begin(),
                            one_hot_encode(features.most_frequent_protocol(), protocol_vocab_, protocol_vec_size_).end());

    // 2. most_freq_conn_state
    encoded_features.insert(encoded_features.end(),
                            one_hot_encode(features.most_frequent_connection_state(), conn_state_vocab_, conn_state_vec_size_).begin(),
                            one_hot_encode(features.most_frequent_connection_state(), conn_state_vocab_, conn_state_vec_size_).end());

    // 3. top_proto_1
    std::vector<std::pair<std::string, int>> sorted_protocols(features.protocol_counts.begin(), features.protocol_counts.end());
    std::sort(sorted_protocols.begin(), sorted_protocols.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::string top_proto_1 = sorted_protocols.empty() ? "other" : sorted_protocols[0].first;
    encoded_features.insert(encoded_features.end(),
                            one_hot_encode(top_proto_1, protocol_vocab_, protocol_vec_size_).begin(),
                            one_hot_encode(top_proto_1, protocol_vocab_, protocol_vec_size_).end());

    // 4, 5, 6. service_1, service_2, service_3
    int service_count = 0;
    for (const auto& service : features.services_used) {
        encoded_features.insert(encoded_features.end(),
                                one_hot_encode(service, service_vocab_, service_vec_size_).begin(),
                                one_hot_encode(service, service_vocab_, service_vec_size_).end());
        if (++service_count >= 3) break;
    }
    // Pad with 'other' encoding if less than 3 services
    while (service_count < 3) {
        encoded_features.insert(encoded_features.end(),
                                one_hot_encode("other", service_vocab_, service_vec_size_).begin(),
                                one_hot_encode("other", service_vocab_, service_vec_size_).end());
        service_count++;
    }

    // 7. top_user_agent_1
    std::vector<std::pair<std::string, int>> sorted_user_agents(features.http_user_agent_counts.begin(), features.http_user_agent_counts.end());
    std::sort(sorted_user_agents.begin(), sorted_user_agents.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::string top_user_agent_1 = sorted_user_agents.empty() ? "Unknown" : sorted_user_agents[0].first;
    encoded_features.insert(encoded_features.end(),
                            one_hot_encode(top_user_agent_1, user_agent_vocab_, user_agent_vec_size_).begin(),
                            one_hot_encode(top_user_agent_1, user_agent_vocab_, user_agent_vec_size_).end());

    // 8. first_seen (assuming it's stored as a string in NodeFeatures for now)
    // You might need a way to access the temporal information if it's in a separate struct.
    // For simplicity, I'll encode it as if it were a string "2025-05-29 19:04:05"
    // If it's already a numerical timestamp, just cast to float.
    std::string first_seen_str = "0.0"; // Replace with actual logic if available
    encoded_features.push_back(std::stof(first_seen_str));

    // 9. last_seen (similarly, assuming string)
    std::string last_seen_str = "0.0"; // Replace with actual logic if available
    encoded_features.push_back(std::stof(last_seen_str));

    return encoded_features;
}