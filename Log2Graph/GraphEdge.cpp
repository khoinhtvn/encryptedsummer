//
// Created by lu on 5/28/25.
//
//

#include "includes/GraphEdge.h"
#include "includes/EdgeFeatureEncoder.h"
#include <sstream>
#include <iomanip>

std::string GraphEdge::to_dot_string_encoded() const {
    std::stringstream ss;
    ss << "  \"" << escape_dot_string(source) << "\" -> \"" << escape_dot_string(target) << "\" [";

    std::vector<std::string> feature_names = EdgeFeatureEncoder::get_feature_names();
    for (size_t i = 0; i < encoded_features.size(); ++i) {
        if (i != 0) {
            ss << ",";
        }
        if (i < feature_names.size()) {
            ss << feature_names[i] << "=" << static_cast<int>(encoded_features[i]);
        } else {
            ss << "feature_" << i << "=" << static_cast<int>(encoded_features[i]);
        }
    }

    ss << "];\n";
    return ss.str();
}

std::string GraphEdge::to_dot_string() const {
    std::stringstream ss;
    ss << "  \"" << escape_dot_string(source) << "\" -> \"" << escape_dot_string(target) <<
          "\" [";

    if (attributes.count("label")) {
        ss << "label=\"" << escape_dot_string(attributes.at("label")) << "\"";
    } else {
        ss << "label=\"" << escape_dot_string(relationship) << "\"";
    }

    for (const auto &attr_pair : attributes) {
        if (attr_pair.first != "label") {
            ss << ", " << escape_dot_string(attr_pair.first) << "=\"" << escape_dot_string(attr_pair.second)
               << "\"";
        }
    }

    std::vector<std::string> feature_names = EdgeFeatureEncoder::get_feature_names();
    for (size_t i = 0; i < encoded_features.size(); ++i) {
        if (i < feature_names.size()) {
            ss << ", " << feature_names[i] << "=" << static_cast<int>(encoded_features[i]);
        } else {
            // Handle potential mismatch in size (shouldn't happen if encoder is consistent)
            ss << ", feature_" << i << "=" << static_cast<int>(encoded_features[i]);
        }
    }

    ss << "];\n";
    return ss.str();
}

std::string GraphEdge::escape_dot_string(const std::string &str) {
    std::string result = "";
    for (char c : str) {
        if (c == '"') {
            result += "\\\"";
        } else if (c == '\\') {
            result += "\\\\";
        } else {
            result += c;
        }
    }
    return result;
}