//
// Created by lu on 6/2/25.
//

#ifndef AGGREGATEDGRAPHEDGE_H
#define AGGREGATEDGRAPHEDGE_H

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * @brief Represents an aggregated edge in the network traffic graph, connecting two GraphNodes,
 * with aggregation based on connection type (protocol, service, destination port).
 */
class AggregatedGraphEdge {
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
     * @brief The protocol of this connection type.
     */
    std::string protocol;
    /**
     * @brief The service of this connection type.
     */
    std::string service;
    /**
     * @brief The destination port of this connection type.
     */
    std::string dst_port;

    // --- Aggregated Numerical Features (Sums) ---
    long long total_orig_bytes;
    long long total_resp_bytes;
    long long total_orig_pkts;
    long long total_resp_pkts;
    long long total_orig_ip_bytes;
    long long total_resp_ip_bytes;

    /**
     * @brief Aggregated encoded features (e.g., sum or mean of individual encoded feature vectors).
     */
    std::vector<float> aggregated_encoded_features;

    /**
     * @brief Timestamp of the first seen connection contributing to this aggregation.
     */
    std::chrono::system_clock::time_point first_seen;
    /**
     * @brief Timestamp of the last seen connection contributing to this aggregation.
     */
    std::chrono::system_clock::time_point last_seen;

    /**
     * @brief Total number of connections aggregated into this edge of this type.
     */
    size_t connection_count;

    /**
     * @brief Constructor for the AggregatedGraphEdge.
     * @param src The identifier of the source node.
     * @param tgt The identifier of the target node.
     * @param proto The protocol of the connection type.
     * @param service The service of the connection type.
     * @param dport The destination port of the connection type.
     */
    AggregatedGraphEdge(const std::string &src, const std::string &tgt,
                        const std::string &proto, const std::string &service, const std::string &dport);

    /**
     * @brief Updates the aggregated edge with new connection data.
     * @param raw_feature_map The raw feature map of the new connection.
     * @param encoded_features The encoded features of the new connection.
     */
    void update(const std::unordered_map<std::string, std::string> &raw_feature_map,
                const std::vector<float> &encoded_features);

    /**
     * @brief Escapes special characters in a string for DOT format.
     * @param str The input string to escape.
     * @return The escaped string.
     */
    static std::string escape_dot_string(const std::string &str);

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

    /**
     * @brief Generates a DOT format string representation of the aggregated edge.
     * @return A string in DOT format representing the aggregated edge.
     */
    std::string to_dot_string() const;

    /**
     * @brief Generates a DOT format string representation with aggregated details.
     * @return A string in DOT format representing the aggregated edge.
     */
    std::string to_dot_string_aggregated() const;
};

#endif // AGGREGATEDGRAPHEDGE_H