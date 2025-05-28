/**
 * @file FeatureEncoder.cpp
 * @brief Implementation of the FeatureEncoder class.
 */

#include "includes/FeatureEncoder.h"
#include <cmath>
#include <ctime>
#include <stdexcept>
#include <sstream>


FeatureEncoder::FeatureEncoder() : protocol_map({{"unknown_transport", 0}, {"tcp", 1}, {"udp", 2}, {"icmp", 3}}),
                                   conn_state_map({
                                       {"SF", 0}, {"S1", 0}, {"REJ", 1}, {"RSTO", 1}, {"RSTR", 1}, {"RSTOS0", 1},
                                       {"RSTRH", 1},
                                       {"S0", 2}, {"S2", 2}, {"S3", 2}, {"SH", 3}, {"SHR", 3}, {"OTH", 4},
                                       {"UNKNOWN", 4}
                                   }),
                                   NUM_CONN_STATE_CATEGORIES(5),
                                   service_map({{"http", 0}, {"ftp", 1}, {"ssh", 2}, {"dns", 3}, {"UNKNOWN", 4}}),
                                   user_agent_map({
                                       {"Chrome", 0}, {"Firefox", 1}, {"Safari", 2}, {"Edge", 3}, {"Opera", 4},
                                       {"Bot", 5}, {"Unknown", 6}
                                   }),
                                   ssl_version_map({
                                       {"", 0}, {"TLSv10", 1}, {"TLSv11", 2}, {"TLSv12", 3}, {"TLSv13", 4},
                                       {"SSLv3", 5}, {"UNKNOWN", 6}
                                   }),
                                   ssl_cipher_map({
                                       {"", 0}, {"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", 1},
                                       {"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", 2},
                                       {"TLS_RSA_WITH_AES_128_CBC_SHA", 3}, {"UNKNOWN", 4}
                                   }),

                                   feature_dimension(
                                       protocol_map.size() + // protocol one-hot
                                       2 + // timestamp (sin, cos)
                                       1 + // src_port (categorized)
                                       1 + // dst_port (categorized)
                                       NUM_CONN_STATE_CATEGORIES + // connection state one-hot
                                       2 + // local_orig and local_resp (binary)
                                       1 + // history length (normalized)
                                       6 + // byte and packet features (normalized)
                                       service_map.size() + // service one-hot
                                       user_agent_map.size() + // http_user_agent one-hot
                                       1 + // http_version (normalized)
                                       1 + // http_status_code (categorized)
                                       ssl_version_map.size() + // ssl_version one-hot
                                       ssl_cipher_map.size() + // ssl_cipher one-hot
                                       1 + // ssl_curve (presence)
                                       1 + // ssl_server_name (presence)
                                       1 + // ssl_resumed (binary)
                                       1 + // ssl_last_alert (presence)
                                       1 + // ssl_next_protocol (presence)
                                       1 + // ssl_established (binary)
                                       1 + // ssl_history (presence)
                                       1 + // ssl_cert_chain_fps (presence)
                                       1 + // ssl_client_cert_chain_fps (presence)
                                       1 // ssl_sni_matches_cert (binary)
                                   ) {
}

size_t FeatureEncoder::get_feature_dimension() const {
    return feature_dimension;
}

float FeatureEncoder::normalize(float value, float min_val, float max_val) {
    if (max_val == min_val) return 0.5f;
    return (value - min_val) / (max_val - min_val);
}

std::vector<float> FeatureEncoder::one_hot(int value, int num_classes) {
    std::vector<float> encoding(num_classes, 0.0f);
    if (value >= 0 && value < num_classes) {
        encoding[value] = 1.0f;
    }
    return encoding;
}

std::vector<float> FeatureEncoder::encode_timestamp_cyclic(const std::string &timestamp) {
    try {
        double ts = std::stod(timestamp);
        const double seconds_per_day = 86400;
        double day_progress = fmod(ts, seconds_per_day) / seconds_per_day;

        float sin_encoding = sin(2 * M_PI * day_progress);
        float cos_encoding = cos(2 * M_PI * day_progress);

        return {sin_encoding, cos_encoding};
    } catch (...) {
        return {0.0f, 1.0f}; // Default on error
    }
}

float FeatureEncoder::encode_port(int port) {
    if (port <= 1023) return 0.0f; // Well-known ports
    if (port <= 49151) return 0.5f; // Registered ports
    return 1.0f; // Dynamic/private ports
}

float FeatureEncoder::normalize_size(size_t size) {
    return std::log1p(static_cast<float>(size)) / std::log1p(1e6f); // Logarithmic scaling
}

float FeatureEncoder::encode_http_version(const std::string &version) {
    if (version.empty()) return 0.0f;
    try {
        return std::stof(version) / 3.0f; // Normalize to a small range
    } catch (...) {
        return 0.0f;
    }
}

float FeatureEncoder::encode_http_status_code(const std::string &code_str) {
    if (code_str.empty()) return 0.0f;
    try {
        int code = std::stoi(code_str);
        if (code >= 100 && code < 200) return 0.1f;
        if (code >= 200 && code < 300) return 0.3f;
        if (code >= 300 && code < 400) return 0.5f;
        if (code >= 400 && code < 500) return 0.7f;
        if (code >= 500 && code < 600) return 0.9f;
        return 0.0f;
    } catch (...) {
        return 0.0f;
    }
}

std::vector<float> FeatureEncoder::encode_features(const std::unordered_map<std::string, std::string> &attrs) {
    std::vector<float> features;

    // Protocol (one-hot)
    auto protocol_it = protocol_map.find(attrs.at("protocol"));
    int protocol_code = (protocol_it != protocol_map.end()) ? protocol_it->second : protocol_map["unknown_transport"];
    auto protocol_encoding = one_hot(protocol_code, protocol_map.size());
    features.insert(features.end(), protocol_encoding.begin(), protocol_encoding.end());

    // Timestamp (cyclic)
    auto ts_encoding = encode_timestamp_cyclic(attrs.at("timestamp"));
    features.insert(features.end(), ts_encoding.begin(), ts_encoding.end());

    // Source and Destination Ports (categorized)
    features.push_back(encode_port(std::stoi(attrs.at("src_port"))));
    features.push_back(encode_port(std::stoi(attrs.at("dst_port"))));

    // Connection State (one-hot)
    auto conn_state_it = conn_state_map.find(attrs.at("conn_state"));
    int conn_state_code = (conn_state_it != conn_state_map.end()) ? conn_state_it->second : conn_state_map["UNKNOWN"];
    auto conn_state_encoding = one_hot(conn_state_code, NUM_CONN_STATE_CATEGORIES);
    features.insert(features.end(), conn_state_encoding.begin(), conn_state_encoding.end());

    // Local Origin and Response (binary)
    features.push_back(attrs.at("local_orig") == "true" ? 1.0f : 0.0f);
    features.push_back(attrs.at("local_resp") == "true" ? 1.0f : 0.0f);

    // History (length normalized)
    features.push_back(normalize(attrs.at("history").length(), 0, 50.0f)); // Arbitrary max length

    // Bytes and Packets (normalized)
    features.push_back(normalize_size(std::stoul(attrs.at("orig_bytes"))));
    features.push_back(normalize_size(std::stoul(attrs.at("resp_bytes"))));
    features.push_back(normalize_size(std::stoul(attrs.at("orig_pkts"))));
    features.push_back(normalize_size(std::stoul(attrs.at("resp_pkts"))));
    features.push_back(normalize_size(std::stoul(attrs.at("orig_ip_bytes"))));
    features.push_back(normalize_size(std::stoul(attrs.at("resp_ip_bytes"))));

    // Service (one-hot)
    auto service_it = service_map.find(attrs.at("service"));
    int service_code = (service_it != service_map.end()) ? service_it->second : service_map["UNKNOWN"];
    auto service_encoding = one_hot(service_code, service_map.size());
    features.insert(features.end(), service_encoding.begin(), service_encoding.end());

    // HTTP User Agent (one-hot - simple categorization)
    std::string user_agent = attrs.count("http_user_agent") ? attrs.at("http_user_agent") : "Unknown";
    std::string user_agent_category = "Unknown";
    if (user_agent.find("Chrome") != std::string::npos) user_agent_category = "Chrome";
    else if (user_agent.find("Firefox") != std::string::npos) user_agent_category = "Firefox";
    else if (user_agent.find("Safari") != std::string::npos) user_agent_category = "Safari";
    else if (user_agent.find("Edge") != std::string::npos) user_agent_category = "Edge";
    else if (user_agent.find("Opera") != std::string::npos) user_agent_category = "Opera";
    else if (user_agent.find("Bot") != std::string::npos) user_agent_category = "Bot";
    auto user_agent_it = user_agent_map.find(user_agent_category);
    int user_agent_code = (user_agent_it != user_agent_map.end()) ? user_agent_it->second : user_agent_map["Unknown"];
    auto user_agent_encoding = one_hot(user_agent_code, user_agent_map.size());
    features.insert(features.end(), user_agent_encoding.begin(), user_agent_encoding.end());

    // HTTP Version (normalized)
    features.push_back(encode_http_version(attrs.count("http_version") ? attrs.at("http_version") : ""));

    // HTTP Status Code (categorized)
    features.push_back(encode_http_status_code(attrs.count("http_status_code") ? attrs.at("http_status_code") : ""));

    // SSL Related Features
    std::string ssl_version = attrs.count("ssl_version") ? attrs.at("ssl_version") : "";
    auto ssl_version_it = ssl_version_map.find(ssl_version);
    int ssl_version_code = (ssl_version_it != ssl_version_map.end())
                               ? ssl_version_it->second
                               : ssl_version_map["UNKNOWN"];
    auto ssl_version_encoding = one_hot(ssl_version_code, ssl_version_map.size());
    features.insert(features.end(), ssl_version_encoding.begin(), ssl_version_encoding.end());

    std::string ssl_cipher = attrs.count("ssl_cipher") ? attrs.at("ssl_cipher") : "";
    auto ssl_cipher_it = ssl_cipher_map.find(ssl_cipher);
    int ssl_cipher_code = (ssl_cipher_it != ssl_cipher_map.end()) ? ssl_cipher_it->second : ssl_cipher_map["UNKNOWN"];
    auto ssl_cipher_encoding = one_hot(ssl_cipher_code, ssl_cipher_map.size());
    features.insert(features.end(), ssl_cipher_encoding.begin(), ssl_cipher_encoding.end());

    // Presence of other SSL features
    features.push_back(attrs.count("ssl_curve") && !attrs.at("ssl_curve").empty() ? 1.0f : 0.0f);
    features.push_back(attrs.count("ssl_server_name") && !attrs.at("ssl_server_name").empty() ? 1.0f : 0.0f);
    features.push_back(attrs.at("ssl_resumed") == "true" ? 1.0f : 0.0f);
    features.push_back(attrs.count("ssl_last_alert") && !attrs.at("ssl_last_alert").empty() ? 1.0f : 0.0f);
    features.push_back(attrs.count("ssl_next_protocol") && !attrs.at("ssl_next_protocol").empty() ? 1.0f : 0.0f);
    features.push_back(attrs.at("ssl_established") == "true" ? 1.0f : 0.0f);
    features.push_back(attrs.count("ssl_history") && !attrs.at("ssl_history").empty() ? 1.0f : 0.0f);
    features.push_back(attrs.count("ssl_cert_chain_fps") && !attrs.at("ssl_cert_chain_fps").empty() ? 1.0f : 0.0f);
    features.push_back(attrs.count("ssl_client_cert_chain_fps") && !attrs.at("ssl_client_cert_chain_fps").empty()
                           ? 1.0f
                           : 0.0f);
    features.push_back(attrs.at("ssl_sni_matches_cert") == "true" ? 1.0f : 0.0f);

    return features;
}

std::vector<std::string> FeatureEncoder::get_feature_names() {
    std::vector<std::string> names;
    // Protocol features
    names.push_back("protocol_UNKNOWN");
    names.push_back("protocol_TCP");
    names.push_back("protocol_UDP");
    names.push_back("protocol_ICMP");
    // Timestamp features
    names.push_back("timestamp_sin");
    names.push_back("timestamp_cos");
    // Port features
    names.push_back("src_port_type");
    names.push_back("dst_port_type");
    // Connection state features
    names.push_back("conn_state_successful");
    names.push_back("conn_state_rejected_reset");
    names.push_back("conn_state_partial");
    names.push_back("conn_state_suspicious");
    names.push_back("conn_state_other");
    // Local origin and response
    names.push_back("local_orig");
    names.push_back("local_resp");
    // History feature
    names.push_back("history_length");
    // Byte and packet features
    names.push_back("orig_bytes");
    names.push_back("resp_bytes");
    names.push_back("orig_pkts");
    names.push_back("resp_pkts");
    names.push_back("orig_ip_bytes");
    names.push_back("resp_ip_bytes");
    // Service features
    names.push_back("service_HTTP");
    names.push_back("service_FTP");
    names.push_back("service_SSH");
    names.push_back("service_DNS");
    names.push_back("service_UNKNOWN");
    // HTTP User Agent features
    names.push_back("http_user_agent_Chrome");
    names.push_back("http_user_agent_Firefox");
    names.push_back("http_user_agent_Safari");
    names.push_back("http_user_agent_Edge");
    names.push_back("http_user_agent_Opera");
    names.push_back("http_user_agent_Bot");
    names.push_back("http_user_agent_Unknown");
    // HTTP Version
    names.push_back("http_version");
    // HTTP Status Code
    names.push_back("http_status_code");
    // SSL Features

    names.push_back("ssl_version_UNKNOWN");
    names.push_back("ssl_version_TLSv10");
    names.push_back("ssl_version_TLSv11");
    names.push_back("ssl_version_TLSv12");
    names.push_back("ssl_version_TLSv13");
    names.push_back("ssl_version_SSLv3");
    names.push_back("ssl_version_UNKNOWN_MAPPED"); // For the "UNKNOWN" key in the map

    names.push_back("ssl_cipher_UNKNOWN");
    names.push_back("ssl_cipher_TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256");
    names.push_back("ssl_cipher_TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384");
    names.push_back("ssl_cipher_TLS_RSA_WITH_AES_128_CBC_SHA");
    names.push_back("ssl_cipher_UNKNOWN_MAPPED"); // For the "UNKNOWN" key in the map

    names.push_back("ssl_curve_present");
    names.push_back("ssl_server_name_present");
    names.push_back("ssl_resumed");
    names.push_back("ssl_last_alert_present");
    names.push_back("ssl_next_protocol_present");
    names.push_back("ssl_established");
    names.push_back("ssl_history_present");
    names.push_back("ssl_cert_chain_fps_present");
    names.push_back("ssl_client_cert_chain_fps_present");
    names.push_back("ssl_sni_matches_cert");

    return names;
}

std::string FeatureEncoder::get_feature_name(size_t index) {
    auto names = get_feature_names();
    if (index >= names.size()) {
        throw std::out_of_range("Feature index out of range");
    }
    return names[index];
}
