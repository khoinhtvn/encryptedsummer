/**
 * @file FeatureEncoder.h
 * @brief Defines the FeatureEncoder class, which encodes network traffic features into a numerical vector.
 *
 * This class is designed for use in network traffic analysis or intrusion detection systems.
 * It handles various data types commonly found in network traffic, including:
 * - Protocol
 * - HTTP Method
 * - User Agent
 * - HTTP Status Code and Message
 * - Timestamp
 * - Port Numbers
 * - String Lengths (Host, URI)
 * - Body Lengths (Request, Response)
 * - HTTP Version
 *
 * The encoding process involves:
 * - One-hot encoding for categorical features.
 * - Normalization for numerical features.
 * - Cyclic encoding and feature extraction for timestamps.
 * - Categorization of HTTP status codes.
 */

// Created by lu on 5/9/25.

#ifndef FEATUREENCODER_H
#define FEATUREENCODER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdexcept>

/**
 * @class FeatureEncoder
 * @brief Encodes network traffic features into a numerical vector representation.
 *
 * The FeatureEncoder class transforms raw network traffic data into a format suitable for
 * machine learning models.  It provides methods to encode various attributes of network
 * traffic, such as protocol, HTTP method, and user agent, into numerical features.
 */
class FeatureEncoder {
private:
    /**
     * @brief Maps protocol names to integer codes for one-hot encoding.
     *
     * The "UNKNOWN" protocol is included to handle unrecognized protocols.
     */
    std::unordered_map<std::string, int> protocol_map = {
        {"unknown_transport", 0}, {"tcp", 1}, {"udp", 2}, {"icmp", 3}
    };

    /**
 * @brief Encodes network connection attributes into a vector of float features.
 *
 * This method takes a map of connection attributes (e.g., protocol, timestamp, ports, bytes) and
 * transforms them into a standardized vector of float features for machine learning models.
 *
 * @param attrs A map of connection attributes, where each key is a string representing the attribute name
 *               (e.g., "protocol", "timestamp", "src_port", "conn_state") and each value is the attribute value as a string.
 * @return A vector of float values representing the encoded features.
 */
    std::unordered_map<std::string, int> conn_state_map = {
        // Successful connections
        {"SF", 0},   // Normal establishment and termination
        {"S1", 0},   // Established but not terminated

        // Rejected/Reset connections
        {"REJ", 1},  // Connection attempt rejected
        {"RSTO", 1}, // Originator aborted
        {"RSTR", 1}, // Responder sent RST
        {"RSTOS0", 1}, // SYN followed by RST
        {"RSTRH", 1}, // SYN ACK followed by RST

        // Partial connections
        {"S0", 2},   // Attempt seen, no reply
        {"S2", 2},   // Established, originator close attempt
        {"S3", 2},   // Established, responder close attempt

        // Suspicious patterns
        {"SH", 3},   // SYN followed by FIN
        {"SHR", 3},  // SYN ACK followed by FIN

        // Other
        {"OTH", 4},  // No SYN seen
        {"UNKNOWN", 4} // Unknown
    };

    const int NUM_CONN_STATE_CATEGORIES = 5;
    // Then use one-hot encoding for these 5 categories instead of 14
    /**
 * @brief Maps network service names to integer codes for feature encoding.
 *
 * This map assigns a unique integer ID to known network services (e.g., HTTP, FTP, SSH).
 * It is used during feature encoding to one-hot encode the "service" attribute
 * of a network connection. The "UNKNOWN" key is used as a fallback for services
 * not explicitly listed in the map.
 */
    std::unordered_map<std::string, int> service_map = {
        {"http", 0},
        {"ftp", 1},
        {"ssh", 2},
        {"dns", 3},
        {"UNKNOWN", 4}
    };

    /**
     * @brief Maps HTTP method names to integer codes for one-hot encoding.
     *
     * "UNKNOWN" handles unrecognized methods.
     */
    std::unordered_map<std::string, int> method_map = {
        {"GET", 0}, {"POST", 1}, {"PUT", 2}, {"DELETE", 3}, {"HEAD", 4}, {"OPTIONS", 5}, {"UNKNOWN", 6}
    };

    /**
     * @brief Maps common user agent strings (or categories) to integer codes.
     *
     * Used for one-hot encoding. "Bot" is included for automated clients,
     * and "Unknown" for user agents not in the list.
     */
    std::unordered_map<std::string, int> user_agent_map = {
        {"Chrome", 0}, {"Firefox", 1}, {"Safari", 2}, {"Edge", 3}, {"Opera", 4}, {"Bot", 5}, {"Unknown", 6}
    };

    /**
     * @brief Maps HTTP status message categories to integer codes for one-hot encoding.
     */
    std::unordered_map<std::string, int> status_msg_map = {
        {"OK", 0}, {"Created", 1}, {"Accepted", 2}, {"Not Found", 3},
        {"Forbidden", 4}, {"Server Error", 5}, {"Other", 6}
    };

    size_t feature_dimension = 0; ///< The total number of features in the encoded vector.

public:
    /**
     * @brief Constructor for the FeatureEncoder class.
     *
     * Calculates the total feature dimension based on the sizes of the encoding maps
     * and the number of scalar features.
     */
    FeatureEncoder() {
        // Calculate fixed output dimension:
        feature_dimension =
                protocol_map.size() + // protocol one-hot
                2 + // timestamp (sin, cos)
                1 + // src_port
                1 + // dst_port
                NUM_CONN_STATE_CATEGORIES + // connection state one-hot
                2 + // local_orig and local_resp (binary)
                1 + // history length
                6 + // byte and packet features
                service_map.size(); // service one-hot
    }

    /**
     * @brief Gets the dimension of the encoded feature vector.
     *
     * @return The total number of features.
     */
    size_t get_feature_dimension() const {
        return feature_dimension;
    }

    /**
     * @brief Normalizes a value to the range [0, 1].
     *
     * @param value The value to normalize.
     * @param min_val The minimum value in the original range.
     * @param max_val The maximum value in the original range.
     * @return The normalized value.  Returns 0.5f if max_val == min_val.
     */
    float normalize(float value, float min_val, float max_val) {
        if (max_val == min_val) return 0.5f;
        return (value - min_val) / (max_val - min_val);
    }

    /**
     * @brief Performs one-hot encoding of an integer value.
     *
     * @param value The integer value to encode.
     * @param num_classes The total number of classes.
     * @return A vector of floats representing the one-hot encoding.
     */
    std::vector<float> one_hot(int value, int num_classes) {
        std::vector<float> encoding(num_classes, 0.0f);
        if (value >= 0 && value < num_classes) {
            encoding[value] = 1.0f;
        }
        return encoding;
    }

    /**
     * @brief Encodes a timestamp into cyclic features (sin and cos) to capture seasonality.
     *
     * @param timestamp The timestamp string to encode.
     * @return A vector containing the sine and cosine encodings of the timestamp.
     * Returns {0.0f, 1.0f} on error.
     */
    std::vector<float> encode_timestamp_cyclic(const std::string &timestamp) {
        try {
            double ts = std::stod(timestamp);
            const double seconds_per_day = 86400;
            double day_progress = fmod(ts, seconds_per_day) / seconds_per_day;

            float sin_encoding = sin(2 * M_PI * day_progress);
            float cos_encoding = cos(2 * M_PI * day_progress);

            return {sin_encoding, cos_encoding};
        } catch (...) {
            return {0.0f, 1.0f}; // Default
        }
    }

    /**
     * @brief Encodes a timestamp string into several separate features.
     *
     * @param timestamp The timestamp string.
     * @return A vector of floats representing the hour, day of week, day of month, and month.
     */
    std::vector<float> encode_timestamp_features(const std::string &timestamp) {
        try {
            time_t ts = static_cast<time_t>(std::stod(timestamp));
            struct tm *timeinfo = std::gmtime(&ts);

            return {
                normalize(timeinfo->tm_hour, 0, 23), // Hour of day
                normalize(timeinfo->tm_wday, 0, 6), // Day of week
                normalize(timeinfo->tm_mday, 1, 31), // Day of month
                normalize(timeinfo->tm_mon, 0, 11) // Month
            };
        } catch (...) {
            return {0.5f, 0.5f, 0.5f, 0.5f}; // Default
        }
    }

    /**
     * @brief Encodes a port number into a category.
     *
     * @param port The port number.
     * @return 0.0f for well-known ports, 0.5f for registered ports, and 1.0f for dynamic ports.
     */
    float encode_port(int port) {
        // Well-known ports: 0-1023, registered: 1024-49151, dynamic: 49152-65535
        if (port <= 1023) return 0.0f; // Well-known
        if (port <= 49151) return 0.5f; // Registered
        return 1.0f; // Dynamic
    }

    /**
     * @brief Encodes the length of a string.
     *
     * @param s The string.
     * @return The normalized string length (0-1000).
     */
    float encode_string_length(const std::string &s) {
        return normalize(s.length(), 0, 1000);
    }

    /**
     * @brief Encodes a body length.
     *
     * @param len The body length.
     * @return The normalized body length (0-10MB).
     */
    float encode_body_length(size_t len) {
        return normalize(len, 0, 10 * 1024 * 1024);
    }

    /**
     * @brief Encodes an HTTP status code into a category.
     *
     * @param code The HTTP status code.
     * @return A vector representing the status code category:
     * {1,0,0,0} for Success (200s), {0,1,0,0} for Redirection (300s),
     * {0,0,1,0} for Client Error (400s), {0,0,0,1} for Server Error (500s),
     * or {0,0,0,0} for Unknown.
     */
    std::vector<float> encode_status_code(int code) {
        // Categorize HTTP status codes
        if (code >= 200 && code < 300) return {1.0f, 0.0f, 0.0f, 0.0f}; // Success
        if (code >= 300 && code < 400) return {0.0f, 1.0f, 0.0f, 0.0f}; // Redirection
        if (code >= 400 && code < 500) return {0.0f, 0.0f, 1.0f, 0.0f}; // Client error
        if (code >= 500) return {0.0f, 0.0f, 0.0f, 1.0f}; // Server error
        return {0.0f, 0.0f, 0.0f, 0.0f}; // Unknown
    }

    /**
     * @brief Normalizes a size value (e.g., bytes or packets) to a range of 0 to 1.
     *
     * This function takes a size value (e.g., bytes or packet count) and scales it to a range
     * of 0 to 1 using a logarithmic scale. This approach effectively handles a wide range of sizes,
     * preventing extremely large values from skewing the data distribution.
     *
     * @param size The size value to be normalized.
     * @return A float value representing the normalized size between 0 and 1.
     */
    float normalize_size(size_t size) {
        // Use logarithmic scaling to handle a wide range of sizes
        return std::log1p(static_cast<float>(size)) / std::log1p(1e6f); // Assuming 1MB as upper bound
    }

    /**
     * @brief Encodes a set of features into a numerical vector.
     *
     * @param attrs A map of attribute names and their corresponding string values.
     * @return A vector of floats representing the encoded features.
     */
    std::vector<float> encode_features(const std::unordered_map<std::string, std::string> &attrs) {
        std::vector<float> features;

        // Protocol (one-hot encoded)
        auto protocol_it = protocol_map.find(attrs.at("protocol"));
        int protocol_code = (protocol_it != protocol_map.end())
                                ? protocol_it->second
                                : protocol_map["unknown_transport"];
        auto protocol_encoding = one_hot(protocol_code, protocol_map.size());
        features.insert(features.end(), protocol_encoding.begin(), protocol_encoding.end());

        // Timestamp (cyclic)
        auto ts_encoding = encode_timestamp_cyclic(attrs.at("timestamp"));
        features.insert(features.end(), ts_encoding.begin(), ts_encoding.end());

        // Source and Destination Ports (categorized)
        features.push_back(encode_port(std::stoi(attrs.at("src_port"))));
        features.push_back(encode_port(std::stoi(attrs.at("dst_port"))));

        // Connection State (one-hot encoded)
        auto conn_state_it = conn_state_map.find(attrs.at("conn_state"));
        int conn_state_code = (conn_state_it != conn_state_map.end())
                                  ? conn_state_it->second
                                  : NUM_CONN_STATE_CATEGORIES - 1;
        auto conn_state_encoding = one_hot(conn_state_code, NUM_CONN_STATE_CATEGORIES);
        features.insert(features.end(), conn_state_encoding.begin(), conn_state_encoding.end());

        // Local Origin and Response (binary)
        features.push_back(attrs.at("local_orig") == "true" ? 1.0f : 0.0f);
        features.push_back(attrs.at("local_resp") == "true" ? 1.0f : 0.0f);

        // History (length normalized)
        features.push_back(static_cast<float>(attrs.at("history").length()) / 10.0f);

        // Bytes and Packets (normalized)
        features.push_back(normalize_size(std::stoul(attrs.at("orig_bytes"))));
        features.push_back(normalize_size(std::stoul(attrs.at("resp_bytes"))));
        features.push_back(normalize_size(std::stoul(attrs.at("orig_pkts"))));
        features.push_back(normalize_size(std::stoul(attrs.at("resp_pkts"))));
        features.push_back(normalize_size(std::stoul(attrs.at("orig_ip_bytes"))));
        features.push_back(normalize_size(std::stoul(attrs.at("resp_ip_bytes"))));

        // Service (one-hot encoded)
        auto service_it = service_map.find(attrs.at("service"));
        int service_code = (service_it != service_map.end()) ? service_it->second : service_map["UNKNOWN"];
        auto service_encoding = one_hot(service_code, service_map.size());
        features.insert(features.end(), service_encoding.begin(), service_encoding.end());
        return features;
    }

    /**
     * @brief Returns the names of the features used in the encoded feature vector.
     *
     * These feature names correspond to the features encoded in the encode_features method,
     * and their order must match the encoded vector.
     *
     * @return A vector of strings representing the names of the encoded features.
     */
    static std::vector<std::string> get_feature_names() {
        return {
            // Protocol features
            "protocol_UNKNOWN", "protocol_TCP", "protocol_UDP", "protocol_ICMP",

            // Timestamp features
            "timestamp_sin", "timestamp_cos",

            // Port features
            "src_port_type", "dst_port_type",

            // Connection state features
            "conn_state_successful",
            "conn_state_rejected_reset",
            "conn_state_partial",
            "conn_state_suspicious",
            "conn_state_other",
            // Local origin and response (binary)
            "local_orig", "local_resp",

            // History feature
            "history_length",

            // Byte and packet features
            "orig_bytes", "resp_bytes",
            "orig_pkts", "resp_pkts",
            "orig_ip_bytes", "resp_ip_bytes",

            // Service features
            "service_HTTP", "service_FTP", "service_SSH",
            "service_DNS", "service_UNKNOWN"
        };
    }

    /**
     * @brief Get the name for a specific feature index
     * @param index The feature index
     * @return Name of the feature at that index
     * @throws std::out_of_range if index is invalid
     */
    static std::string get_feature_name(size_t index) {
        auto names = get_feature_names();
        if (index >= names.size()) {
            throw std::out_of_range("Feature index out of range");
        }
        return names[index];
    }
};

#endif // FEATUREENCODER_H
