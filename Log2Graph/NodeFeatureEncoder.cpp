#include "includes/NodeFeatureEncoder.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <cmath>

const std::vector<std::string> PROTOCOLS = {"tcp", "udp", "icmp", "other"};
const std::vector<std::string> CONN_STATES = {"S0", "S1", "SF", "REJ", "RSTO", "RSTR", "OTH", "SH", "SHR", "RSTOS0", "RSTRH", "other"};
const std::vector<std::string> SERVICES = {"-", "http", "ssl", "dns", "ftp", "ssh", "rdp", "smb", "other"};
const std::vector<std::string> USER_AGENTS = {"Mozilla", "Chrome", "Safari", "Edge", "curl", "Wget", "Python", "Java", "Unknown"};
const std::vector<std::string> BOOLEANS = {"false", "true"};
const std::vector<std::string> HTTP_VERSIONS = {"HTTP/1.0", "HTTP/1.1", "HTTP/2", "other"};
const std::vector<std::string> SSL_VERSIONS = {"SSLv3", "TLSv10", "TLSv11", "TLSv12", "TLSv13", "other"};

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

    // Initialize user agent vocabulary (including "Unknown")
    for (size_t i = 0; i < USER_AGENTS.size(); ++i) {
        user_agent_vocab_[USER_AGENTS[i]] = i;
    }
    user_agent_vec_size_ = USER_AGENTS.size();

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

    // Initialize SSL/TLS version vocabulary
    for (size_t i = 0; i < SSL_VERSIONS.size(); ++i) {
        ssl_version_vocab_[SSL_VERSIONS[i]] = i;
    }
    ssl_version_vec_size_ = SSL_VERSIONS.size();
}

std::vector<float> NodeFeatureEncoder::one_hot_encode(const std::string& value,
                                                     const std::map<std::string, int>& vocabulary,
                                                     size_t vector_size) const {
    std::vector<float> encoded_vector(vector_size, 0.0f);
    auto it = vocabulary.find(value);
    if (it != vocabulary.end()) {
        encoded_vector[it->second] = 1.0f;
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
    std::string top_user_agent_1 = sorted_user_agents.empty() ? "Unknown" : sorted_user_agents[0].first;
    temp_encoding = one_hot_encode(top_user_agent_1, user_agent_vocab_, user_agent_vec_size_);
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
    names.push_back("most_freq_proto");
    names.push_back("most_freq_conn_state");
    names.push_back("top_proto_1");
    names.push_back("service_1");
    names.push_back("service_2");
    names.push_back("service_3");
    names.push_back("top_user_agent_1");
    names.push_back("first_seen_seconds");
    names.push_back("last_seen_seconds");
    names.push_back("outgoing_connection_ratio");
    names.push_back("incoming_connection_ratio");
    names.push_back("unique_remote_ports_connected_to");
    names.push_back("unique_local_ports_used");
    names.push_back("unique_remote_ports_connected_from");
    names.push_back("unique_local_ports_listening_on");
    names.push_back("ever_connected_to_privileged_port");
    names.push_back("ever_listened_on_privileged_port");
    names.push_back("unique_http_versions_used");
    names.push_back("most_freq_http_version"); // Base name, will be expanded in to_dot_string
    names.push_back("unique_http_status_codes_seen");
    names.push_back("has_http_error_4xx");
    names.push_back("has_http_error_5xx");
    names.push_back("unique_ssl_versions_used");
    names.push_back("most_freq_ssl_version"); // Base name, will be expanded in to_dot_string
    names.push_back("unique_ssl_ciphers_used");
    names.push_back("has_ssl_resumption");
    names.push_back("has_ssl_server_name");
    names.push_back("has_ssl_history");
    return names;
}