//
// Created by lu on 5/7/25.
//

#ifndef GRAPHEDGE_H
#define GRAPHEDGE_H
#include <unordered_map>
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

    /**
     * @brief Constructor for the GraphEdge.
     * @param src The identifier of the source node.
     * @param tgt The identifier of the target node.
     * @param rel The type of relationship between the nodes.
     */
    GraphEdge(const std::string &src, const std::string &tgt,
              const std::string &rel)
        : source(src), target(tgt), relationship(rel) {
    }
};
#endif //GRAPHEDGE_H
