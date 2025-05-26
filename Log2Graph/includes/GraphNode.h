//
// Created by lu on 5/7/25.
//

#ifndef GRAPHNODE_H
#define GRAPHNODE_H
#include <atomic>
#include <map>
#include <queue>
#include <unordered_map>
/**
 * @brief Represents a node in the network traffic graph.
 *
 * Each `GraphNode` corresponds to an entity in the network, such as a host, service, or domain.
 * It stores information about the node's identity, type, attributes, and various statistical features
 * derived from the network traffic.
 */
class GraphNode {
public:
    /**
     * @brief Unique identifier for the node.
     */
    std::string id;
    /**
     * @brief Type of the node (e.g., "host", "service", "domain").
     */
    std::string type;
    /**
     * @brief Additional attributes associated with the node.
     */
    std::unordered_map<std::string, std::string> attributes;

    /**
     * @brief Structure to hold various statistical features of the node.
     */
    struct {
        /**
         * @brief Total degree of the node (number of connected edges).
         */
        std::atomic<int> degree{0};
        /**
         * @brief In-degree of the node (number of incoming edges).
         */
        std::atomic<int> in_degree{0};
        /**
         * @brief Out-degree of the node (number of outgoing edges).
         */
        std::atomic<int> out_degree{0};
        /**
          * @brief Counts of different protocols associated with the node's connections.
          * The key is the protocol name, and the value is the count.
          */
        std::map<std::string, int> protocol_counts;
        /**
         * @brief Score indicating the level of activity of the node.
         */
        std::atomic<double> activity_score{0.0};
    } features;

    /**
     * @brief Structure to hold temporal features related to the node's connections over time.
     */
    struct {
        /**
         * @brief Number of connections initiated or received by the node in the last minute.
         */
        std::atomic<int> connections_last_minute{0};
        /**
         * @brief Number of connections initiated or received by the node in the last hour.
         */
        std::atomic<int> connections_last_hour{0};
        /**
         * @brief Timestamp of when the monitoring of this node started.
         */
        std::chrono::system_clock::time_point monitoring_start;
        /**
         * @brief Total number of connections associated with this node since monitoring started.
         */
        std::atomic<int> total_connections{0};
        /**
         * @brief Vector storing timestamps of recent connections for more fine-grained analysis.
         */
        std::vector<std::chrono::system_clock::time_point> recent_connections;
        /**
         * @brief Queue to store timestamps of connections within the last minute for efficient counting.
         */
        std::queue<std::chrono::system_clock::time_point> minute_window;
        /**
         * @brief Queue to store timestamps of connections within the last hour for efficient counting.
         */
        std::queue<std::chrono::system_clock::time_point> hour_window;
        /**
         * @brief Mutex to protect access to the `minute_window` and `hour_window` queues, ensuring thread safety.
         */
        mutable std::mutex window_mutex;
    } temporal;

    /**
     * @brief Constructor for the GraphNode.
     * @param id The unique identifier of the node.
     * @param type The type of the node (e.g., "host", "service").
     */
    GraphNode(const std::string &id, const std::string &type)
        : id(id), type(type) {
        temporal.monitoring_start = std::chrono::system_clock::now();
    }

    /**
     * @brief Updates the connection-related features of the node.
     *
     * This method increments the degree counters and updates the protocol counts
     * based on a new connection involving this node.
     *
     * @param protocol The protocol of the connection (e.g., "tcp", "udp").
     * @param is_outgoing True if the connection is outgoing from this node, false otherwise.
     */
    void update_connection_features(const std::string &protocol, bool is_outgoing);

    /**
     * @brief Removes timestamps of old connections from the `recent_connections` vector.
     *
     * This method helps to keep the `recent_connections` vector manageable and relevant
     * for short-term analysis.
     */
    void cleanup_old_connections();

    /**
     * @brief Calculates an anomaly score for the node based on its current features.
     * @return A double representing the anomaly score.
     */
    double calculate_anomaly_score() const;

    /**
     * @brief Removes outdated timestamps from the minute and hour time windows.
     *
     * This method ensures that the connection counts for the last minute and last hour
     * are accurate by discarding timestamps that fall outside these windows.
     */
    void cleanup_time_windows();

    /**
     * @brief Gets the number of connections associated with this node in the last minute.
     * @return The number of connections in the last minute.
     */
    int get_connections_last_minute() const;

    /**
     * @brief Gets the number of connections associated with this node in the last hour.
     * @return The number of connections in the last hour.
     */
    int get_connections_last_hour() const;
};
#endif //GRAPHNODE_H
