/**
 * @file EdgeFeatureEncoder.cpp
 * @brief Implementation of the EdgeFeatureEncoder class.
 */

#include "includes/EdgeFeatureEncoder.h"
#include <cmath>
#include <ctime>
#include <stdexcept>
#include <sstream>

EdgeFeatureEncoder::EdgeFeatureEncoder() : protocol_map({{"unknown_transport", 0}, {"tcp", 1}, {"udp", 2}, {"icmp", 3}}),
                                     conn_state_map({
                                         {"SF", 0}, {"S1", 0}, // Successful/Established
                                         {"REJ", 1}, {"RSTO", 1}, {"RSTR", 1}, {"RSTOS0", 1},
                                         {"RSTRH", 1},           // Rejected/Reset
                                         {"S0", 2}, {"S2", 2}, {"S3", 2}, {"SH", 2}, {"SHR", 2}, {"OTH", 2},
                                         {"UNKNOWN", 2}          // Partial/Other/Unknown
                                     }),
                                     NUM_CONN_STATE_CATEGORIES(3),
                                     ssl_version_map({
                                         {"", 0}, {"TLSv10", 1}, {"TLSv11", 2}, {"TLSv12", 3}, {"TLSv13", 4},
                                         {"SSLv3", 5}, {"UNKNOWN", 6}
                                     }),
                                     NUM_SSL_VERSION_CATEGORIES(ssl_version_map.size()),
                                     user_agent_map({
                                         {"Chrome", 0}, {"Firefox", 1}, {"Safari", 2}, {"Edge", 3}, {"Opera", 4},
                                         {"Bot", 5}, {"Unknown", 6}
                                     }),
                                     NUM_USER_AGENT_CATEGORIES(user_agent_map.size()),
                                     feature_dimension(
                                         protocol_map.size() +       // protocol one-hot
                                         NUM_CONN_STATE_CATEGORIES + // connection state one-hot (reduced)
                                         NUM_SSL_VERSION_CATEGORIES + // ssl_version one-hot
                                         NUM_USER_AGENT_CATEGORIES   // user_agent one-hot (simplified)
                                     ) {}

size_t EdgeFeatureEncoder::get_feature_dimension() const {
    return feature_dimension;
}

std::vector<float> EdgeFeatureEncoder::one_hot(int value, int num_classes) {
    std::vector<float> encoding(num_classes, 0.0f);
    if (value >= 0 && value < num_classes) {
        encoding[value] = 1.0f;
    }
    return encoding;
}

std::vector<float> EdgeFeatureEncoder::encode_features(const std::unordered_map<std::string, std::string> &attrs) {
    std::vector<float> features;

    // Protocol (one-hot)
    if (attrs.count("protocol")) {
        auto protocol_it = protocol_map.find(attrs.at("protocol"));
        int protocol_code = (protocol_it != protocol_map.end()) ? protocol_it->second : protocol_map["unknown_transport"];
        auto protocol_encoding = one_hot(protocol_code, protocol_map.size());
        features.insert(features.end(), protocol_encoding.begin(), protocol_encoding.end());
    } else {
        auto protocol_encoding = one_hot(protocol_map["unknown_transport"], protocol_map.size());
        features.insert(features.end(), protocol_encoding.begin(), protocol_encoding.end());
    }

    // Connection State (one-hot - reduced to 3 categories)
    int conn_state_code = 2; // Default to "Partial/Other/Unknown"
    if (attrs.count("conn_state")) {
        auto conn_state_it = conn_state_map.find(attrs.at("conn_state"));
        if (conn_state_it != conn_state_map.end()) {
            conn_state_code = conn_state_it->second;
        }
    }
    auto conn_state_encoding = one_hot(conn_state_code, NUM_CONN_STATE_CATEGORIES);
    features.insert(features.end(), conn_state_encoding.begin(), conn_state_encoding.end());

    // SSL Version (one-hot)
    int ssl_version_code = ssl_version_map[""]; // Default to empty/no SSL
    if (attrs.count("ssl_version")) {
        auto ssl_version_it = ssl_version_map.find(attrs.at("ssl_version"));
        if (ssl_version_it != ssl_version_map.end()) {
            ssl_version_code = ssl_version_it->second;
        } else {
            ssl_version_code = ssl_version_map["UNKNOWN"];
        }
    }
    auto ssl_version_encoding = one_hot(ssl_version_code, NUM_SSL_VERSION_CATEGORIES);
    features.insert(features.end(), ssl_version_encoding.begin(), ssl_version_encoding.end());

    // User Agent (one-hot - simplified categorization)
    int user_agent_code = user_agent_map["Unknown"];
    if (attrs.count("http_user_agent")) {
        std::string user_agent = attrs.at("http_user_agent");
        std::string user_agent_category = "Unknown";
        if (user_agent.find("Chrome") != std::string::npos) user_agent_category = "Chrome";
        else if (user_agent.find("Firefox") != std::string::npos) user_agent_category = "Firefox";
        else if (user_agent.find("Safari") != std::string::npos) user_agent_category = "Safari";
        else if (user_agent.find("Edge") != std::string::npos) user_agent_category = "Edge";
        else if (user_agent.find("Opera") != std::string::npos) user_agent_category = "Opera";
        else if (user_agent.find("Bot") != std::string::npos) user_agent_category = "Bot";
        auto user_agent_it = user_agent_map.find(user_agent_category);
        if (user_agent_it != user_agent_map.end()) {
            user_agent_code = user_agent_it->second;
        }
    }
    auto user_agent_encoding = one_hot(user_agent_code, NUM_USER_AGENT_CATEGORIES);
    features.insert(features.end(), user_agent_encoding.begin(), user_agent_encoding.end());

    return features;
}

std::vector<std::string> EdgeFeatureEncoder::get_feature_names() {
    std::vector<std::string> names;
    // Protocol features
    names.push_back("protocol_UNKNOWN");
    names.push_back("protocol_TCP");
    names.push_back("protocol_UDP");
    names.push_back("protocol_ICMP");
    // Connection state features (reduced)
    names.push_back("conn_state_successful");
    names.push_back("conn_state_rejected_reset");
    names.push_back("conn_state_partial_other_unknown");
    // SSL Version features
    names.push_back("ssl_version_NONE");
    names.push_back("ssl_version_TLSv10");
    names.push_back("ssl_version_TLSv11");
    names.push_back("ssl_version_TLSv12");
    names.push_back("ssl_version_TLSv13");
    names.push_back("ssl_version_SSLv3");
    names.push_back("ssl_version_UNKNOWN");
    // User Agent features
    names.push_back("user_agent_Chrome");
    names.push_back("user_agent_Firefox");
    names.push_back("user_agent_Safari");
    names.push_back("user_agent_Edge");
    names.push_back("user_agent_Opera");
    names.push_back("user_agent_Bot");
    names.push_back("user_agent_Unknown");
    return names;
}

std::string EdgeFeatureEncoder::get_feature_name(size_t index) {
    auto names = get_feature_names();
    if (index >= names.size()) {
        throw std::out_of_range("Feature index out of range");
    }
    return names[index];
}