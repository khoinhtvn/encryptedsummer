//
// Created by lu on 5/28/25.
//

#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include <atomic>        // For atomic operations on primitive types
#include <map>           // For ordered key-value pairs (e.g., protocol counts)
#include <unordered_map> // For unordered key-value pairs (e.g., attributes)
#include <vector>        // For dynamic arrays (not directly used in the node itself but might be in related structures)
#include <chrono>        // For time-related functionalities (e.g., tracking first/last seen)
#include <mutex>         // For synchronizing access to shared resources (prevents race conditions)
#include <queue>         // For managing time windows of connections
#include <set>           // For storing unique elements (e.g., protocols used)
#include <string>        // For representing textual data (IDs, types, attributes)
#include <sstream>       // For building strings (e.g., in to_dot_string)
#include <iomanip>       // For formatting output (e.g., time)

class NodeFeatureEncoder;

class GraphNode {
public:
    /**
     * @brief Structure to hold various features of the network node.
     */
    struct NodeFeatures {
        std::atomic<uint32_t> degree{0};                                     ///< Total number of connections to/from this node.
        std::atomic<uint32_t> in_degree{0};                                  ///< Number of incoming connections.
        std::atomic<uint32_t> out_degree{0};                                 ///< Number of outgoing connections.
        std::map<std::string, int> protocol_counts;                          ///< Count of connections per protocol.
        std::atomic<double> activity_score{0.0};                              ///< A score indicating the node's activity level.
        std::atomic<int> total_connections_initiated{0};                     ///< Total number of connections initiated by this node.
        std::atomic<int> total_connections_received{0};                     ///< Total number of connections received by this node.
        std::set<std::string> protocols_used;                                ///< Set of unique protocols used by this node.
        std::set<std::string> remote_ports_connected_to;                    ///< Set of unique remote ports this node connected to.
        std::set<std::string> local_ports_used;                             ///< Set of unique local ports used by this node.
        std::set<std::string> remote_ports_connected_from;                  ///< Set of unique remote ports that connected to this node.
        std::set<std::string> local_ports_listening_on;                   ///< Set of unique local ports this node was listening on.
        std::map<std::string, int> connection_state_counts;                  ///< Count of connections per connection state (e.g., ESTABLISHED).
        std::atomic<bool> ever_local_originated{false};                      ///< True if this node ever initiated a local connection.
        std::atomic<bool> ever_local_responded{false};                     ///< True if this node ever responded to a local connection.
        std::atomic<long long> total_orig_bytes{0};                           ///< Total number of bytes sent by this node.
        std::atomic<long long> total_resp_bytes{0};                           ///< Total number of bytes received by this node.
        std::atomic<long long> total_orig_pkts{0};                            ///< Total number of packets sent by this node.
        std::atomic<long long> total_resp_pkts{0};                            ///< Total number of packets received by this node.
        std::set<std::string> services_used;                                ///< Set of unique services used by this node (e.g., http, dns).
        std::map<std::string, int> http_user_agent_counts;                   ///< Count of connections per HTTP user agent.
        std::set<std::string> http_versions_used;                             ///< Set of unique HTTP versions used.
        std::map<int, int> http_status_code_counts;                         ///< Count of connections per HTTP status code.
        std::set<std::string> ssl_versions_used;                             ///< Set of unique SSL/TLS versions used.
        std::set<std::string> ssl_ciphers_used;                             ///< Set of unique SSL/TLS ciphers used.
        std::atomic<bool> ever_ssl_curve_present{false};                     ///< True if an SSL/TLS elliptic curve was ever present.
        std::atomic<bool> ever_ssl_server_name_present{false};                ///< True if an SSL/TLS server name was ever present.
        std::atomic<int> ssl_resumption_count{0};                           ///< Count of SSL/TLS session resumptions.
        std::atomic<bool> ever_ssl_last_alert_present{false};                ///< True if an SSL/TLS alert was ever present.
        std::set<std::string> ssl_next_protocols_used;                      ///< Set of unique SSL/TLS next protocols used (e.g., h2).
        std::atomic<int> ssl_established_count{0};                          ///< Count of successful SSL/TLS handshakes.
        std::atomic<bool> ever_ssl_history_present{false};                   ///< True if SSL/TLS history was ever present.
        std::atomic<double> avg_packet_size_sent{0.0};                       ///< Average size of packets sent.
        std::atomic<double> avg_packet_size_received{0.0};                    ///< Average size of packets received.

        // Historical Aggregation Features
        std::atomic<long long> historical_total_orig_bytes{0};              ///< Total original bytes aggregated over time.
        std::atomic<long long> historical_total_resp_bytes{0};              ///< Total response bytes aggregated over time.
        std::map<std::string, int> historical_protocol_counts;             ///< Aggregated counts of protocols over time.
        std::atomic<int> historical_total_connections{0};                  ///< Total number of connections aggregated over time.

        double outgoing_connection_ratio() const;
        double incoming_connection_ratio() const;
        std::string most_frequent_protocol() const;
        size_t unique_remote_ports_connected_to() const;
        size_t unique_local_ports_used() const;
        std::string most_frequent_connection_state() const;
        bool connected_to_privileged_port() const;
        bool listened_on_privileged_port() const;
        long long total_bytes_sent() const { return total_orig_bytes; }
        long long total_bytes_received() const { return total_resp_bytes; }
        bool used_ssl_tls() const;
        const std::set<std::string>& get_ssl_versions_used() const { return ssl_versions_used; }
        const std::map<std::string, int>& get_http_user_agent_counts() const { return http_user_agent_counts; }
    };

    /**
     * @brief Structure to hold temporal features of the network node.
     */
    struct TemporalFeatures {
        std::atomic<int> connections_last_minute{0}; ///< Number of connections seen in the last minute.
        std::atomic<int> connections_last_hour{0};   ///< Number of connections seen in the last hour.
        std::chrono::system_clock::time_point monitoring_start; ///< Time when monitoring of this node started.
        std::atomic<int> total_connections{0};       ///< Total number of connections this node has been involved in.
        mutable std::mutex window_mutex;             ///< Mutex to protect the minute_window and hour_window queues.
        std::queue<std::chrono::system_clock::time_point> minute_window; ///< Queue to track connection times for the last minute.
        std::queue<std::chrono::system_clock::time_point> hour_window;   ///< Queue to track connection times for the last hour.
        std::string first_seen;                      ///< Timestamp of when this node was first observed.
        std::string last_seen;                       ///< Timestamp of when this node was last observed in a connection.
    };

    std::string id;                                                                 ///< Unique identifier of the node (e.g., IP address).
    std::string type;                                                               ///< Type of the node (e.g., host, network).
private:
    mutable std::mutex node_mutex;                                                  ///< Mutex to protect the entire GraphNode object for thread-safe access.
    std::unordered_map<std::string, std::string> attributes;                      ///< Additional attributes of the node.
    NodeFeatures features;                                                          ///< Statistical features of the node based on its traffic.
    TemporalFeatures temporal;                                                      ///< Time-based features of the node.
    std::chrono::system_clock::time_point last_connection_time;                    ///< Timestamp of the last connection involving this node.
    std::atomic<uint64_t> connection_count{0};                                      ///< Total number of connections associated with this node.

    // Static instance of the NodeFeatureEncoder
    static const NodeFeatureEncoder node_feature_encoder;

public:
    /**
     * @brief Constructor for the GraphNode.
     * @param id The unique identifier of the node.
     * @param type The type of the node.
     */
    GraphNode(const std::string &id, const std::string &type);

    /**
     * @brief Updates the features of the node based on a new connection.
     * @param is_outgoing True if the connection was initiated by this node, false otherwise.
     * @param connection_attributes Map of attributes associated with the connection.
     */
    void update_connection_features(bool is_outgoing,
                                     const std::unordered_map<std::string, std::string> &connection_attributes);

    /**
     * @brief Removes connection timestamps that are outside the current time windows.
     */
    void cleanup_time_windows();

    /**
     * @brief Gets the number of connections seen in the last minute.
     * @return The count of connections in the last minute.
     */
    int get_connections_last_minute() const;

    /**
     * @brief Gets the number of connections seen in the last hour.
     * @return The count of connections in the last hour.
     */
    int get_connections_last_hour() const;

    /**
     * @brief Gets the ID of the node.
     * @return The node's ID.
     */
    std::string get_id() const { return id; }

    /**
     * @brief Generates a DOT format string representation of the node, including its features.
     * @return A string in DOT format representing the node.
     */
    std::string to_dot_string() const;
    /**
     * @brief Generates a DOT format string representation of the node, including its encoded features.
     * @return A string in DOT format representing the node.
     */
    std::string to_dot_string_encoded() const;
    /**
     * @brief Escapes special characters in a string for DOT format.
     * @param str The string to escape.
     * @return The escaped string.
     */
    static std::string escape_dot_string(const std::string &str);

    /**
     * @brief Aggregates historical connection data into the node's features.
     * @param orig_bytes Number of original bytes from the connection.
     * @param resp_bytes Number of response bytes from the connection.
     * @param protocol The protocol of the connection.
     */
    void aggregate_historical_data(long long orig_bytes, long long resp_bytes, const std::string& protocol);

    void increment_degree();
    void decrement_degree();
    void increment_in_degree();
    void decrement_in_degree();
    void increment_out_degree();
    void decrement_out_degree();
    void reset_degree();
    void reset_in_degree();
    void reset_out_degree();
};

#endif // GRAPHNODE_H