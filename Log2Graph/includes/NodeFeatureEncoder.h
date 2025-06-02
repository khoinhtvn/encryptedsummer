#ifndef NODE_FEATURE_ENCODER_H
#define NODE_FEATURE_ENCODER_H

#include <vector>
#include <string>
#include <map>

#include "GraphNode.h"

class NodeFeatureEncoder {
public:
    static const std::vector<std::string> PROTOCOLS;
    static const std::vector<std::string> CONN_STATES;
    static const std::vector<std::string> SERVICES;
    static const std::vector<std::string> USER_AGENTS;
    static const std::vector<std::string> BOOLEANS;
    static const std::vector<std::string> HTTP_VERSIONS;
    static const std::vector<std::string> SSL_VERSIONS;

    static std::vector<std::string> feature_names_;

    NodeFeatureEncoder(); // Constructor

    size_t protocol_vec_size() const { return protocol_vec_size_; }
    size_t conn_state_vec_size() const { return conn_state_vec_size_; }
    size_t service_vec_size() const { return service_vec_size_; }
    size_t user_agent_vec_size() const { return user_agent_vec_size_; }
    size_t bool_vec_size() const { return bool_vec_size_; }
    size_t http_version_vec_size() const { return http_version_vec_size_; }
    size_t ssl_version_vec_size() const { return ssl_version_vec_size_; }

    std::vector<float> one_hot_encode(const std::string& value,
                                     const std::map<std::string, int>& vocabulary,
                                     size_t vector_size) const;

    std::pair<double, double> encode_hour_minute_to_circle(long long timestamp) const;

    std::vector<float> encode_node_features(const struct GraphNode::NodeFeatures& features, const struct GraphNode::TemporalFeatures& temporal) const;
    std::vector<std::string> get_feature_names_base() const;
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

    std::map<std::string, int> http_version_vocab_;
    size_t http_version_vec_size_;

    std::map<std::string, int> ssl_version_vocab_;
    size_t ssl_version_vec_size_;
};

#endif // NODE_FEATURE_ENCODER_H