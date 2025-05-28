/**
 * @file FeatureEncoder.h
 * @brief Defines the FeatureEncoder class, which encodes network traffic features into a numerical vector.
 *
 * This class is designed for use in network traffic analysis or intrusion detection systems.
 * It handles various data types commonly found in network traffic, including:
 * - Protocol (TCP, UDP, ICMP)
 * - Timestamp of the connection
 * - Source and Destination Port numbers
 * - Connection State (e.g., established, rejected)
 * - Flags indicating local origin/response
 * - Length of the connection history string
 * - Number of original and response bytes and packets
 * - Network Service (e.g., HTTP, FTP, SSH)
 * - SSL/TLS related features (version, cipher, server name, etc.)
 * - HTTP related features (method, host, URI, user agent, status code, etc.)
 *
 * The encoding process involves:
 * - One-hot encoding for categorical features (protocol, connection state, service, HTTP method, user agent, SSL version, SSL cipher).
 * - Cyclic encoding for timestamps to capture temporal patterns.
 * - Categorization for port numbers (well-known, registered, dynamic).
 * - Normalization (logarithmic) for continuous numerical features (bytes, packets).
 * - Binary encoding for boolean flags (local origin/response, SSL resumed/established, SNI matches cert).
 * - Direct inclusion or simple normalization for other features (history length, HTTP version, status code).
 * - Categorical encoding for SSL related strings (cipher, version).
 */

// Created by lu on 5/9/25.

#ifndef FEATUREENCODER_H
#define FEATUREENCODER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

class FeatureEncoder {
private:
    // --- Encoding Maps for Categorical Features ---
    std::unordered_map<std::string, int> protocol_map;
    std::unordered_map<std::string, int> conn_state_map;
    const int NUM_CONN_STATE_CATEGORIES;
    std::unordered_map<std::string, int> service_map;
    std::unordered_map<std::string, int> method_map;
    std::unordered_map<std::string, int> user_agent_map;
    std::unordered_map<std::string, int> ssl_version_map;
    std::unordered_map<std::string, int> ssl_cipher_map;

    size_t feature_dimension; ///< The total number of features in the encoded vector.

    // --- Utility Encoding Functions ---
    float normalize(float value, float min_val, float max_val);

    std::vector<float> one_hot(int value, int num_classes);

    std::vector<float> encode_timestamp_cyclic(const std::string &timestamp);

    float encode_port(int port);

    float normalize_size(size_t size);

    float encode_http_version(const std::string &version);

    float encode_http_status_code(const std::string &code_str);

public:
    /**
     * @brief Constructor for the FeatureEncoder class.
     *
     * Initializes the encoding maps and calculates the total feature dimension.
     */
    FeatureEncoder();

    /**
     * @brief Gets the dimension of the encoded feature vector.
     *
     * @return The total number of features.
     */
    size_t get_feature_dimension() const;

    /**
     * @brief Encodes a set of network connection attributes into a numerical feature vector.
     *
     * @param attrs A map of connection attributes (e.g., protocol, timestamp, ports, bytes, etc.).
     * @return A vector of float values representing the encoded features.
     */
    std::vector<float> encode_features(const std::unordered_map<std::string, std::string> &attrs);

    /**
     * @brief Returns the names of the features used in the encoded feature vector.
     *
     * The order of names corresponds to the order of features in the encoded vector.
     * @return A vector of strings representing the names of the encoded features.
     */
    static std::vector<std::string> get_feature_names();

    /**
     * @brief Get the name for a specific feature index.
     *
     * @param index The feature index.
     * @return Name of the feature at that index.
     * @throws std::out_of_range if index is invalid.
     */
    static std::string get_feature_name(size_t index);
};


#endif // FEATUREENCODER_H
