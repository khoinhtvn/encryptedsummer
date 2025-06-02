//
// Created by lu on 5/7/25.
//

#ifndef GRAPHEDGE_H
#define GRAPHEDGE_H
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include "EdgeFeatureEncoder.h" // Include FeatureEncoder for feature names
#include <sstream>
#include <iomanip>

/**
 * @brief Represents an edge in the network traffic graph, connecting two GraphNodes.
 *
 * Each `GraphEdge` represents a communication or relationship between two network entities.
 * It stores the identifiers of the source and target nodes, the type of relationship,
 * and any additional attributes associated with the connection.
 */
class GraphEdge {
public:
    /**
     * @brief Identifier of the source node.
     */
    std::string source;
    /**
     * @brief Identifier of the target node.
     */
    std::string target;
    /**
     * @brief Type of relationship between the source and target nodes (e.g., "connects_to", "sends_data_to").
     */
    std::string relationship;
    /**
     * @brief Additional attributes associated with the edge.
     */
    std::unordered_map<std::string, std::string> attributes;
    /**
     * @brief Encoded features, to feed to GAT.
     */
    std::vector<float> encoded_features;

    std::chrono::system_clock::time_point last_seen;
    /**
     * @brief Constructor for the GraphEdge.
     * @param src The identifier of the source node.
     * @param tgt The identifier of the target node.
     * @param rel The type of relationship between the nodes.
     */
    GraphEdge(const std::string &src, const std::string &tgt,
              const std::string &rel)
        : source(src), target(tgt), relationship(rel),
          last_seen(std::chrono::system_clock::now())
    { }


    /**
     * @brief Escapes special characters in a string for DOT format.
     * @param str The input string to escape.
     * @return The escaped string.
     */
    static std::string escape_dot_string(const std::string &str) ;

    /**
     * @brief Gets the ID of the source node.
     * @return The source node's ID.
     */
    std::string get_source_node_id() const { return source; }

    /**
     * @brief Gets the ID of the destination node.
     * @return The destination node's ID.
     */
    std::string get_destination_node_id() const { return target; }

    std::string to_dot_string_encoded() const;

    /**
     * @brief Generates a DOT format string representation of the edge, including its attributes and encoded features.
     * @return A string in DOT format representing the edge.
     */
    std::string to_dot_string() const;


};
#endif //GRAPHEDGE_H