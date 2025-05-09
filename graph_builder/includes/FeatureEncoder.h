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
        {"HTTP", 0}, {"HTTPS", 1}, {"FTP", 2}, {"SSH", 3}, {"DNS", 4}, {"UNKNOWN", 5}
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
            protocol_map.size() +  // protocol one-hot
            1 +                    // timestamp
            1 +                    // src_port
            1 +                    // dst_port
            method_map.size() +    // method one-hot
            1 +                    // host length
            1 +                    // uri length
            1 +                    // version
            user_agent_map.size() +  // user_agent one-hot
            1 +                    // request_body_len
            1 +                    // response_body_len
            4 +                    // status_code (4 categories)
            status_msg_map.size();  // status_msg one-hot
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
    std::vector<float> encode_timestamp_cyclic(const std::string& timestamp) {
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
    std::vector<float> encode_timestamp_features(const std::string& timestamp) {
        try {
            time_t ts = static_cast<time_t>(std::stod(timestamp));
            struct tm* timeinfo = std::gmtime(&ts);

            return {
                normalize(timeinfo->tm_hour, 0, 23),    // Hour of day
                normalize(timeinfo->tm_wday, 0, 6),     // Day of week
                normalize(timeinfo->tm_mday, 1, 31),    // Day of month
                normalize(timeinfo->tm_mon, 0, 11)      // Month
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
        if (port <= 1023) return 0.0f;  // Well-known
        if (port <= 49151) return 0.5f; // Registered
        return 1.0f;                   // Dynamic
    }

    /**
     * @brief Encodes the length of a string.
     *
     * @param s The string.
     * @return The normalized string length (0-1000).
     */
    float encode_string_length(const std::string& s) {
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
        if (code >= 500) return {0.0f, 0.0f, 0.0f, 1.0f};            // Server error
        return {0.0f, 0.0f, 0.0f, 0.0f};                        // Unknown
    }

    /**
     * @brief Encodes a set of features into a numerical vector.
     *
     * @param attrs A map of attribute names and their corresponding string values.
     * @return A vector of floats representing the encoded features.
     */
    std::vector<float> encode_features(const std::unordered_map<std::string, std::string>& attrs) {
        std::vector<float> features;

        // Protocol (one-hot encoded)
        auto protocol_it = protocol_map.find(attrs.at("protocol"));
        int protocol_code = (protocol_it != protocol_map.end()) ? protocol_it->second : protocol_map["UNKNOWN"];
        auto protocol_encoding = one_hot(protocol_code, protocol_map.size());
        features.insert(features.end(), protocol_encoding.begin(), protocol_encoding.end());

        // Timestamp (cyclic)
        auto ts_encoding = encode_timestamp_cyclic(attrs.at("timestamp"));
        features.insert(features.end(), ts_encoding.begin(), ts_encoding.end());

        // Source port (categorized)
        int src_port = std::stoi(attrs.at("src_port"));
        features.push_back(encode_port(src_port));

        // Destination port (categorized)
        int dst_port = std::stoi(attrs.at("dst_port"));
        features.push_back(encode_port(dst_port));

        // Method (one-hot encoded)
        auto method_it = method_map.find(attrs.at("method"));
        int method_code = (method_it != method_map.end()) ? method_it->second : method_map["UNKNOWN"];
        auto method_encoding = one_hot(method_code, method_map.size());
        features.insert(features.end(), method_encoding.begin(), method_encoding.end());

        // Host (length normalized)
        features.push_back(encode_string_length(attrs.at("host")));

        // URI (length normalized)
        features.push_back(encode_string_length(attrs.at("uri")));

        // Version (extract HTTP version number)
        float http_version = 1.0f;
        if (attrs.at("version").find("1.1") != std::string::npos) http_version = 1.1f;
        else if (attrs.at("version").find("2.0") != std::string::npos) http_version = 2.0f;
        features.push_back(http_version);

        // User agent (categorized)
        std::string ua = attrs.at("user_agent");
        int ua_code = user_agent_map["Unknown"];
        for (const auto& [key, val] : user_agent_map) {
            if (ua.find(key) != std::string::npos) {
                ua_code = val;
                break;
            }
        }
        auto ua_encoding = one_hot(ua_code, user_agent_map.size());
        features.insert(features.end(), ua_encoding.begin(), ua_encoding.end());

        // Request body length (normalized)
        size_t req_len = std::stoul(attrs.at("request_body_len"));
        features.push_back(encode_body_length(req_len));

        // Response body length (normalized)
        size_t res_len = std::stoul(attrs.at("response_body_len"));
        features.push_back(encode_body_length(res_len));

        // Status code (categorized)
        int status_code = std::stoi(attrs.at("status_code"));
        auto status_code_encoding = encode_status_code(status_code);
        features.insert(features.end(), status_code_encoding.begin(), status_code_encoding.end());

        // Status message (categorized)
        std::string status_msg = attrs.at("status_msg");
        int status_msg_code = status_msg_map["Other"];
        for (const auto& [key, val] : status_msg_map) {
            if (status_msg.find(key) != std::string::npos) {
                status_msg_code = val;
                break;
            }
        }
        auto status_msg_encoding = one_hot(status_msg_code, status_msg_map.size());
        features.insert(features.end(), status_msg_encoding.begin(), status_msg_encoding.end());

        return features;
    }
      /**
     * @brief Get the names of all encoded features in order
     * @return Vector of feature names corresponding to the encoded features
     */
    static std::vector<std::string> get_feature_names() {
        return {
            // Protocol features
            "protocol_HTTP", "protocol_HTTPS", "protocol_FTP",
            "protocol_SSH", "protocol_DNS", "protocol_UNKNOWN",

            // Timestamp features
            "timestamp_sin", "timestamp_cos",

            // Port features
            "src_port_type", "dst_port_type",

            // Method features
            "method_GET", "method_POST", "method_PUT",
            "method_DELETE", "method_HEAD", "method_OPTIONS", "method_UNKNOWN",

            // String length features
            "host_length", "uri_length",

            // Version feature
            "http_version",

            // User agent features
            "ua_Chrome", "ua_Firefox", "ua_Safari",
            "ua_Edge", "ua_Opera", "ua_Bot", "ua_Unknown",

            // Body length features
            "request_body_len", "response_body_len",

            // Status features
            "status_success", "status_redirection",
            "status_client_error", "status_server_error",

            // Status message features
            "statusmsg_OK", "statusmsg_Created", "statusmsg_Accepted",
            "statusmsg_Not_Found", "statusmsg_Forbidden",
            "statusmsg_Server_Error", "statusmsg_Other"
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
