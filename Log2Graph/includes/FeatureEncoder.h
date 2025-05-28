/**
 * @file FeatureEncoder.h
 * @brief Defines the FeatureEncoder class, which encodes network traffic features into a numerical vector.
 *
 * This class is designed for use in network traffic analysis or intrusion detection systems.
 * It encodes a selected set of edge attributes for graph representation.
 * The encoding process involves:
 * - One-hot encoding for categorical features (protocol, connection state, SSL version, user agent).
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
    const int NUM_PROTOCOL_CATEGORIES = protocol_map.size();
    std::unordered_map<std::string, int> conn_state_map;
    const int NUM_CONN_STATE_CATEGORIES;
    std::unordered_map<std::string, int> ssl_version_map;
    const int NUM_SSL_VERSION_CATEGORIES;
    std::unordered_map<std::string, int> user_agent_map;
    const int NUM_USER_AGENT_CATEGORIES;

    size_t feature_dimension; ///< The total number of features in the encoded vector.

    // --- Utility Encoding Functions ---
    std::vector<float> one_hot(int value, int num_classes);

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
     * @param attrs A map of connection attributes.
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