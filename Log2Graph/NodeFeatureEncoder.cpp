#include "includes/NodeFeatureEncoder.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cmath>
#include <iostream>


const std::vector<std::string> NodeFeatureEncoder::PROTOCOLS = {"tcp", "udp", "icmp", "other"};
const std::vector<std::string> NodeFeatureEncoder::CONN_STATES = {"S0", "S1", "SF", "REJ", "RSTO", "RSTR", "OTH", "SH", "SHR", "RSTOS0", "RSTRH", "other"};
const std::vector<std::string> NodeFeatureEncoder::SERVICES = {"-", "http", "ssl", "dns", "ftp", "ssh", "rdp", "smb", "other"};
const std::vector<std::string> NodeFeatureEncoder::USER_AGENTS = {"Firefox", "Chrome", "Safari", "Edge", "curl", "Wget", "Python", "Java", "other"};
const std::vector<std::string> NodeFeatureEncoder::BOOLEANS = {"false", "true"};
const std::vector<std::string> NodeFeatureEncoder::HTTP_VERSIONS = {"1.0", "1.1", "2", "other"};
const std::vector<std::string> NodeFeatureEncoder::SSL_VERSIONS = {"SSLv3", "TLSv10", "TLSv11", "TLSv12", "TLSv13", "other"};


std::vector<std::string> NodeFeatureEncoder::feature_names_;

NodeFeatureEncoder::NodeFeatureEncoder() {
    // Initialize protocol vocabulary
    for (size_t i = 0; i < PROTOCOLS.size(); ++i) {
        protocol_vocab_[PROTOCOLS[i]] = i;
    }
    protocol_vec_size_ = PROTOCOLS.size();

    // Initialize connection state vocabulary
    for (size_t i = 0; i < CONN_STATES.size(); ++i) {
        conn_state_vocab_[CONN_STATES[i]] = i;
    }
    conn_state_vec_size_ = CONN_STATES.size();

    // Initialize service vocabulary
    for (size_t i = 0; i < SERVICES.size(); ++i) {
        service_vocab_[SERVICES[i]] = i;
    }
    service_vec_size_ = SERVICES.size();
    std::cout << "[NodeFeatureEncoder] Service Vocabulary Size: " << service_vocab_.size() << std::endl;
    for (const auto& pair : service_vocab_) {
        std::cout << "[NodeFeatureEncoder]    Service: " << pair.first << " -> " << pair.second << std::endl;
    }

    // Initialize user agent vocabulary (including "Unknown")
    for (size_t i = 0; i < USER_AGENTS.size(); ++i) {
        user_agent_vocab_[USER_AGENTS[i]] = i;
    }
    user_agent_vec_size_ = USER_AGENTS.size();
    std::cout << "[NodeFeatureEncoder] User Agent Vocabulary Size: " << user_agent_vocab_.size() << std::endl;
    for (const auto& pair : user_agent_vocab_) {
        std::cout << "[NodeFeatureEncoder]    User Agent: " << pair.first << " -> " << pair.second << std::endl;
    }

    // Initialize boolean vocabulary
    for (size_t i = 0; i < BOOLEANS.size(); ++i) {
        bool_vocab_[BOOLEANS[i]] = i;
    }
    bool_vec_size_ = BOOLEANS.size();

    // Initialize HTTP version vocabulary
    for (size_t i = 0; i < HTTP_VERSIONS.size(); ++i) {
        http_version_vocab_[HTTP_VERSIONS[i]] = i;
    }
    http_version_vec_size_ = HTTP_VERSIONS.size();
    std::cout << "[NodeFeatureEncoder] HTTP Version Vocabulary Size: " << http_version_vocab_.size() << std::endl;
    for (const auto& pair : http_version_vocab_) {
        std::cout << "[NodeFeatureEncoder]    HTTP Version: " << pair.first << " -> " << pair.second << std::endl;
    }

    // Initialize SSL/TLS version vocabulary
    for (size_t i = 0; i < SSL_VERSIONS.size(); ++i) {
        ssl_version_vocab_[SSL_VERSIONS[i]] = i;
    }
    ssl_version_vec_size_ = SSL_VERSIONS.size();
    std::cout << "[NodeFeatureEncoder] SSL Version Vocabulary Size: " << ssl_version_vocab_.size() << std::endl;
    for (const auto& pair : ssl_version_vocab_) {
        std::cout << "[NodeFeatureEncoder]    SSL Version: " << pair.first << " -> " << pair.second << std::endl;
    }
    feature_names_ = {
        "most_freq_proto",
        "most_freq_conn_state",
        "top_proto_1",
        "service_1", "service_2", "service_3", // Assuming 3 service features
        "top_user_agent_1",
        "first_seen_seconds",
        "last_seen_seconds",
        "outgoing_connection_ratio",
        "incoming_connection_ratio",
        "unique_remote_ports_connected_to",
        "unique_local_ports_used",
        "unique_remote_ports_connected_from",
        "unique_local_ports_listening_on",
        "ever_connected_to_privileged_port",
        "ever_listened_on_privileged_port",
        "unique_http_versions_used",
        "most_freq_http_version",
        "unique_http_status_codes_seen",
        "has_http_error_4xx",
        "has_http_error_5xx",
        "unique_ssl_versions_used",
        "most_freq_ssl_version",
        "unique_ssl_ciphers_used",
        "has_ssl_resumption",
        "has_ssl_server_name",
        "has_ssl_history"
    };
}

std::vector<std::string> NodeFeatureEncoder::get_feature_names() {
    return feature_names_;
}

std::string NodeFeatureEncoder::get_feature_name(size_t index) {
    if (index >= feature_names_.size()) {
        throw std::out_of_range("Feature index out of range");
    }
    return feature_names_[index];
}

std::vector<float> NodeFeatureEncoder::one_hot_encode(const std::string& value,
                                                     const std::map<std::string, int>& vocabulary,
                                                     size_t vector_size) const {
    std::vector<float> encoded_vector(vector_size, 0.0f);
    auto it = vocabulary.find(value);
    if (it != vocabulary.end()) {
        encoded_vector[it->second] = 1.0f;
    } else {
         auto other_it = vocabulary.find("other");
        if (other_it != vocabulary.end()) {
            encoded_vector[other_it->second] = 1.0f;
        } else {
           std::cerr << "[ERROR - NodeFeatureEncoder] 'other' category not found in vocabulary for value: \"" << value << "\"" << std::endl;
            // Handle this critical error appropriately, perhaps by returning the zero vector or throwing an exception.
        }
    }
    return encoded_vector;
}

std::vector<float> NodeFeatureEncoder::encode_node_features(const struct GraphNode::NodeFeatures& features, const struct GraphNode::TemporalFeatures& temporal) const {
    std::vector<float> encoded_features;
    std::vector<float> temp_encoding;

    // 1. most_freq_proto
    temp_encoding = one_hot_encode(features.most_frequent_protocol(), protocol_vocab_, protocol_vec_size_);
    encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());

    // 2. most_freq_conn_state
    temp_encoding = one_hot_encode(features.most_frequent_connection_state(), conn_state_vocab_, conn_state_vec_size_);
    encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());

    // 3. top_proto_1
    std::vector<std::pair<std::string, int>> sorted_protocols(features.protocol_counts.begin(), features.protocol_counts.end());
    std::sort(sorted_protocols.begin(), sorted_protocols.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    std::string top_proto_1 = sorted_protocols.empty() ? "other" : sorted_protocols[0].first;
    temp_encoding = one_hot_encode(top_proto_1, protocol_vocab_, protocol_vec_size_);
    encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());

    // 4, 5, 6. service_1, service_2, service_3
    int service_count = 0;
    for (const auto& service : features.services_used) {
        temp_encoding = one_hot_encode(service, service_vocab_, service_vec_size_);
        encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());
        if (++service_count >= 3) break;
    }
    // Pad with 'other' encoding if less than 3 services
    while (service_count < 3) {
        temp_encoding = one_hot_encode("other", service_vocab_, service_vec_size_);
        encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());
        service_count++;
    }


    // 7. top_user_agent_1
    std::vector<std::pair<std::string, int>> sorted_user_agents(features.http_user_agent_counts.begin(), features.http_user_agent_counts.end());
    std::sort(sorted_user_agents.begin(), sorted_user_agents.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // top_user_agent_1 will now already be a category string because http_user_agent_counts stores categories
    std::string top_user_agent_category = sorted_user_agents.empty() ? "Unknown" : sorted_user_agents[0].first;


    temp_encoding = one_hot_encode(top_user_agent_category, user_agent_vocab_, user_agent_vec_size_);
    encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());


    // 8. first_seen (Convert to integer where possible)
    long long first_seen_int = 0;
    try {
        std::tm t{};
        std::istringstream ss(temporal.first_seen);
        ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
        if (!ss.fail()) {
            std::time_t timeSinceEpoch = mktime(&t);
            if (timeSinceEpoch != -1) {
                first_seen_int = static_cast<long long>(timeSinceEpoch);
            }
        }
    } catch (...) {
        // Keep as 0 if parsing fails
    }
    encoded_features.push_back(static_cast<float>(first_seen_int));

    // 9. last_seen (Convert to integer where possible)
    long long last_seen_int = 0;
    try {
        std::tm t{};
        std::istringstream ss(temporal.last_seen);
        ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
        if (!ss.fail()) {
            std::time_t timeSinceEpoch = mktime(&t);
            if (timeSinceEpoch != -1) {
                last_seen_int = static_cast<long long>(timeSinceEpoch);
            }
        }
    } catch (...) {
        // Keep as 0 if parsing fails
    }
    encoded_features.push_back(static_cast<float>(last_seen_int));

    // 10. outgoing_connection_ratio
    encoded_features.push_back(static_cast<float>(features.outgoing_connection_ratio()));

    // 11. incoming_connection_ratio
    encoded_features.push_back(static_cast<float>(features.incoming_connection_ratio()));

    // 12. unique_remote_ports_connected_to (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.unique_remote_ports_connected_to()));

    // 13. unique_local_ports_used (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.unique_local_ports_used()));

    // 14. unique_remote_ports_connected_from (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.remote_ports_connected_from.size()));

    // 15. unique_local_ports_listening_on (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.local_ports_listening_on.size()));

    // 16. ever_connected_to_privileged_port (Convert to integer boolean)
    encoded_features.push_back(features.connected_to_privileged_port() ? 1.0f : 0.0f);

    // 17. ever_listened_on_privileged_port (Convert to integer boolean)
    encoded_features.push_back(features.listened_on_privileged_port() ? 1.0f : 0.0f);

    // 18. unique_http_versions_used (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.http_versions_used.size()));

    // 19. most_frequent_http_version
    std::string most_freq_http_version = "other";
    int max_http_version_count = 0;
    for (const auto& pair : features.http_version_counts) {
        if (pair.second > max_http_version_count) {
            max_http_version_count = pair.second;
            most_freq_http_version = pair.first;
        }
    }
    // Add this check:
    if (features.http_version_counts.empty()) {
        most_freq_http_version = "other";
    }
   temp_encoding = one_hot_encode(most_freq_http_version, http_version_vocab_, http_version_vec_size_);
    encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());
    // 20. unique_http_status_codes_seen (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.http_status_code_counts.size()));

    // 21. has_http_error_4xx (Convert to integer boolean)
    bool has_4xx = false;
    for (const auto& code : features.http_status_code_counts) {
        if (code.first >= 400 && code.first < 500) {
            has_4xx = true;
            break;
        }
    }
    encoded_features.push_back(has_4xx ? 1.0f : 0.0f);

    // 22. has_http_error_5xx (Convert to integer boolean)
    bool has_5xx = false;
    for (const auto& code : features.http_status_code_counts) {
        if (code.first >= 500 && code.first < 600) {
            has_5xx = true;
            break;
        }
    }
    encoded_features.push_back(has_5xx ? 1.0f : 0.0f);

    // 23. unique_ssl_versions_used (Convert to integer)

    encoded_features.push_back(static_cast<float>(features.ssl_versions_used.size()));

    // 24. most_frequent_ssl_version
    std::string most_freq_ssl_version = "other";
    int max_ssl_version_count = 0;
    for (const auto& version : features.ssl_version_counts) {
        if (version.second > max_ssl_version_count) {
            max_ssl_version_count = version.second;
            most_freq_ssl_version = version.first;
        }
    }
    temp_encoding = one_hot_encode(most_freq_ssl_version, ssl_version_vocab_, ssl_version_vec_size_);
    encoded_features.insert(encoded_features.end(), temp_encoding.begin(), temp_encoding.end());

    // 25. unique_ssl_ciphers_used (Convert to integer)
    encoded_features.push_back(static_cast<float>(features.ssl_ciphers_used.size()));

    // 26. has_ssl_resumption (Convert to integer boolean)
    encoded_features.push_back(features.ssl_resumption_count > 0 ? 1.0f : 0.0f);

    // 27. has_ssl_server_name (Convert to integer boolean)
    encoded_features.push_back(features.ever_ssl_server_name_present.load() ? 1.0f : 0.0f);

    // 28. has_ssl_history (Convert to integer boolean)
    encoded_features.push_back(features.ever_ssl_history_present.load() ? 1.0f : 0.0f);

    return encoded_features;
}

std::vector<std::string> NodeFeatureEncoder::get_feature_names_base() const {
    std::vector<std::string> names;
    names.push_back("most_freq_proto");         // 1
    names.push_back("most_freq_conn_state");    // 2
    names.push_back("top_proto_1");             // 3
    names.push_back("service_1");             // 4
    names.push_back("service_2");             // 5
    names.push_back("service_3");             // 6
    names.push_back("top_user_agent_1");        // 7
    names.push_back("first_seen_seconds");      // 8
    names.push_back("last_seen_seconds");       // 9
    names.push_back("outgoing_connection_ratio");// 10
    names.push_back("incoming_connection_ratio");// 11
    names.push_back("unique_remote_ports_connected_to"); // 12
    names.push_back("unique_local_ports_used");  // 13
    names.push_back("unique_remote_ports_connected_from"); // 14
    names.push_back("unique_local_ports_listening_on");// 15
    names.push_back("ever_connected_to_privileged_port");// 16
    names.push_back("ever_listened_on_privileged_port");// 17
    names.push_back("unique_http_versions_used"); // 18
    names.push_back("most_freq_http_version");    // 19
    names.push_back("unique_http_status_codes_seen");// 20
    names.push_back("has_http_error_4xx");      // 21
    names.push_back("has_http_error_5xx");      // 22
    names.push_back("unique_ssl_versions_used"); // 23
    names.push_back("most_freq_ssl_version");   // 24
    names.push_back("unique_ssl_ciphers_used"); // 25
    names.push_back("has_ssl_resumption");      // 26
    names.push_back("has_ssl_server_name");     // 27
    names.push_back("has_ssl_history");         // 28
    return names;
}