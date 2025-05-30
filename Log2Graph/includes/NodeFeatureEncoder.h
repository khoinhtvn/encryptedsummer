#ifndef NODEFEATUREENCODER_H
#define NODEFEATUREENCODER_H

#include <string>
#include <vector>
#include <map>

#include "GraphNode.h"

class NodeFeatureEncoder {
public:
    NodeFeatureEncoder();

    std::vector<float> one_hot_encode(const std::string& value,
                                         const std::map<std::string, int>& vocabulary,
                                         size_t vector_size) const;

    std::vector<float> encode_node_features(const GraphNode::NodeFeatures &features, const GraphNode::TemporalFeatures &temporal) const;
    std::vector<std::string> get_feature_names_base() const;

    size_t protocol_vec_size() const { return protocol_vec_size_; }
    size_t conn_state_vec_size() const { return conn_state_vec_size_; }
    size_t service_vec_size() const { return service_vec_size_; }
    size_t user_agent_vec_size() const { return user_agent_vec_size_; }
    size_t bool_vec_size() const { return bool_vec_size_; }
    size_t http_version_vec_size() const { return http_version_vec_size_; }
    size_t ssl_version_vec_size() const { return ssl_version_vec_size_; }

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

#endif // NODEFEATUREENCODER_H