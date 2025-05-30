#ifndef NODE_FEATURE_ENCODER_H
#define NODE_FEATURE_ENCODER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>

#include "GraphNode.h"
class NodeFeatureEncoder {
public:
    /**
     * @brief Encodes the features from a GraphNode::NodeFeatures struct
     * into a numerical vector of floats.
     *
     * @param features A const reference to the GraphNode::NodeFeatures struct.
     * @return A vector of floats representing the encoded node features.
     */
    std::vector<float> encode_node_features(const struct GraphNode::NodeFeatures& features) const;

    NodeFeatureEncoder();
private:
    // Vocabularies for one-hot encoding categorical features
    std::map<std::string, int> protocol_vocab_;
    std::map<std::string, int> conn_state_vocab_;
    std::map<std::string, int> service_vocab_;
    std::map<std::string, int> user_agent_vocab_;
    std::map<std::string, int> bool_vocab_; // For boolean features



    // Helper functions for encoding
    std::vector<float> one_hot_encode(const std::string& value,
                                      const std::map<std::string, int>& vocabulary,
                                      size_t vector_size) const;

    // Sizes for one-hot encoded vectors
    size_t protocol_vec_size_;
    size_t conn_state_vec_size_;
    size_t service_vec_size_;
    size_t user_agent_vec_size_;
    size_t bool_vec_size_;
};

#endif // NODE_FEATURE_ENCODER_H