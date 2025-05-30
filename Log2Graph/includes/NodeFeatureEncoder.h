#ifndef NODEFEATUREENCODER_H
#define NODEFEATUREENCODER_H

#include <string>
#include <vector>
#include <map>

#include "GraphNode.h"

class NodeFeatureEncoder {
public:
    NodeFeatureEncoder();
    ~NodeFeatureEncoder() = default;

    /**
     * @brief Encodes a single string value into a one-hot encoded vector.
     *
     * @param value The string value to encode.
     * @param vocabulary A map associating string values with their integer indices.
     * @param vector_size The size of the one-hot encoded vector.
     * @return A vector of floats representing the one-hot encoding of the value.
     * The vector will have a 1.0f at the index corresponding to the value
     * in the vocabulary, and 0.0f elsewhere. If the value is not found,
     * the vector will be all zeros.
     */
    std::vector<float> one_hot_encode(const std::string& value,
                                         const std::map<std::string, int>& vocabulary,
                                         size_t vector_size) const;

    /**
     * @brief Encodes the features of a GraphNode into a vector of floats.
     *
     * This method extracts various features from the NodeFeatures struct and
     * encodes them using one-hot encoding or direct numerical representation.
     * The order of features in the returned vector corresponds to the order
     * of names returned by get_feature_names().
     *
     * @param features The NodeFeatures struct containing the node's attributes.
     * @return A vector of floats representing the encoded features of the node.
     */
    std::vector<float> encode_node_features(const struct GraphNode::NodeFeatures& features) const;

    /**
     * @brief Returns a vector of feature names corresponding to the encoded features.
     *
     * The order of names in the returned vector matches the order of the
     * encoded feature values in the vector returned by encode_node_features().
     *
     * @return A vector of strings containing the names of the encoded features.
     */
    std::vector<std::string> get_feature_names() const;

private:
    std::map<std::string, int> protocol_vocab_;
    size_t protocol_vec_size_;
    std::map<std::string, int> conn_state_vocab_;
    size_t conn_state_vec_size_;
    std::map<std::string, int> service_vocab_;
    size_t service_vec_size_;
    std::map<std::string, int> user_agent_vocab_;
    size_t user_agent_vec_size_;
    std::map<std::string, int> bool_vocab_;
    size_t bool_vec_size_;

    /**
     * @brief Helper function to get the top N key-value pairs from a map sorted by value.
     *
     * @param counts The map to sort and extract from.
     * @param n The number of top elements to retrieve.
     * @return A vector of pairs representing the top N elements.
     */
    std::vector<std::pair<std::string, int>> get_top_n(const std::map<std::string, int>& counts, size_t n) const;
};

#endif // NODEFEATUREENCODER_H