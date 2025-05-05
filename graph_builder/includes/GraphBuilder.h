/**
 * @file GraphBuilder.h
 * @brief Header file for the GraphBuilder class, responsible for constructing the network traffic graph.
 *
 * This file defines the `GraphNode`, `GraphEdge`, `TrafficGraph`, and `GraphBuilder` classes,
 * which together form the data structures and logic for representing and building a graph
 * of network traffic based on Zeek logs.
 */

// Created by lu on 4/25/25.
//

#ifndef GRAPHBUILDER_H
#define GRAPHBUILDER_H
//
// Created by lu on 4/25/25.
//  --> Redundant include

#include <atomic>
#include <map>

//#include "GraphBuilder.h"  --> Self-include is incorrect

#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

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

/**
 * @brief Represents the entire network traffic graph.
 *
 * The `TrafficGraph` class manages a collection of `GraphNode` and `GraphEdge` objects,
 * providing methods to add nodes and edges, access the graph data, and perform
 * graph-based analysis. It also includes a background thread for maintenance tasks.
 */
class TrafficGraph {
    /**
     * @brief Friend class allowing the anomaly detector to access the graph's internal data.
     */
    friend class RealTimeAnomalyDetector;

private:
    /**
     * @brief Map of node IDs to shared pointers of GraphNode objects.
     * Using shared pointers for automatic memory management and shared ownership.
     */
    std::unordered_map<std::string, std::shared_ptr<GraphNode> > nodes;
    /**
     * @brief Vector of shared pointers to GraphEdge objects.
     */
    std::vector<std::shared_ptr<GraphEdge> > edges;
    /**
     * @brief Shared mutex to protect concurrent access to the graph data structures.
     * Allows multiple readers or a single writer.
     */
    mutable std::shared_mutex graph_mutex;
    /**
     * @brief Background thread responsible for performing periodic maintenance tasks on the graph.
     */
    std::thread maintenance_thread;
    /**
     * @brief Atomic boolean flag to control the execution of the maintenance thread.
     */
    std::atomic<bool> running{true};

public:
    /**
     * @brief Constructor for the TrafficGraph.
     *
     * Initializes the maintenance thread, which periodically cleans up the time windows
     * of all nodes in the graph.
     */
    TrafficGraph() {
        maintenance_thread = std::thread([this]() {
            while (running) {
                clean_all_time_windows();
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
        });
    }

    /**
     * @brief Destructor for the TrafficGraph.
     *
     * Stops the maintenance thread gracefully by setting the `running` flag to false
     * and joining the thread.
     */
    ~TrafficGraph() {
        running = false;
        if (maintenance_thread.joinable()) {
            maintenance_thread.join();
        }
    }

    /**
     * @brief Cleans up the minute and hour time windows for all nodes in the graph.
     *
     * This method acquires a shared lock on the graph and then iterates through all
     * nodes, calling their `cleanup_time_windows()` method.
     */
    void clean_all_time_windows() {
        std::shared_lock lock(graph_mutex);
        for (auto &[id, node]: nodes) {
            node->cleanup_time_windows();
        }
    }

    /**
     * @brief Adds a new node to the graph.
     *
     * If a node with the given ID already exists, this method does nothing.
     *
     * @param id The unique identifier of the node to add.
     * @param type The type of the node (e.g., "host", "service").
     */
    void add_node(const std::string &id, const std::string &type);

    /**
     * @brief Adds a new edge to the graph.
     *
     * Creates a new `GraphEdge` object and adds it to the list of edges.
     *
     * @param src The identifier of the source node.
     * @param tgt The identifier of the target node.
     * @param rel The type of relationship between the nodes.
     * @param attrs Optional attributes to associate with the edge. Defaults to an empty map.
     */
    void add_edge(const std::string &src, const std::string &tgt,
                  const std::string &rel, const std::unordered_map<std::string, std::string> &attrs = {});

    /**
     * @brief Retrieves an existing node or creates a new one if it doesn't exist.
     *
     * This method first tries to find a node with the given ID. If found, it returns a
     * reference to it. If not found, it creates a new node with the given ID and type,
     * adds it to the graph, and then returns a reference to the newly created node.
     *
     * @param id The unique identifier of the node.
     * @param type The type of the node.
     * @return A reference to the GraphNode.
     */
    GraphNode &get_or_create_node(const std::string &id, const std::string &type);

    /**
     * @brief Gets a vector of all nodes in the graph.
     *
     * Acquires a shared lock for read access before returning the nodes.
     *
     * @return A vector of shared pointers to all GraphNode objects in the graph.
     */
    std::vector<std::shared_ptr<GraphNode> > get_nodes() const;

    /**
     * @brief Gets a vector of all edges in the graph.
     *
     * Acquires a shared lock for read access before returning the edges.
     *
     * @return A vector of shared pointers to all GraphEdge objects in the graph.
     */
    std::vector<std::shared_ptr<GraphEdge> > get_edges() const;

     /**
     * @brief Finds the connected components of the graph.
     *
     * This method performs a graph traversal (e.g., Depth-First Search or Breadth-First Search)
     * to identify groups of nodes that are connected to each other.
     *
     * @return A vector of vectors of strings, where each inner vector represents a connected component
     * (e.g., by listing the IDs of the nodes in that component).
     */
    std::vector<std::string> find_connected_components() const;

    /**
     * @brief Detects anomalous nodes in the graph based on certain criteria.
     *
     * This method analyzes the features of the nodes (e.g., degree, activity score)
     * to identify nodes that deviate significantly from the norm.
     *
     * @return A vector of strings, where each string is the ID of an anomalous node.
     */
    std::vector<std::string> detect_anomalies() const;

    /**
     * @brief Checks if the graph contains any nodes.
     * @return True if the graph is empty (contains no nodes), false otherwise.
     */
    bool is_empty() const;

    // Add more analysis methods as needed
};

/**
 * @brief Singleton class responsible for building and managing the network traffic graph.
 *
 * The `GraphBuilder` class provides a single point of access to the `TrafficGraph`
 * and offers methods to process raw network traffic data (e.g., from Zeek logs)
 * and add corresponding nodes and edges to the graph. The singleton pattern ensures
 * that only one instance of the graph builder exists throughout the application.
 */
class GraphBuilder {
private:
    /**
     * @brief Static unique pointer to the single instance of GraphBuilder.
     */
    static std::unique_ptr<GraphBuilder> instance;
    /**
     * @brief Static mutex to protect the creation of the singleton instance in a thread-safe manner.
     */
    static std::mutex instance_mutex;

    /**
     * @brief The underlying traffic graph being built and managed.
     */
    TrafficGraph graph;

    /**
     * @brief Private default constructor to enforce the singleton pattern.
     */
    GraphBuilder() = default;

public:
    /**
     * @brief Deleted copy constructor to prevent copying of the singleton instance.
     */
    GraphBuilder(const GraphBuilder &) = delete;

    /**
     * @brief Deleted assignment operator to prevent assignment of the singleton instance.
     * @return void
     */
    GraphBuilder &operator=(const GraphBuilder &) = delete;

    /**
     * @brief Gets the singleton instance of the GraphBuilder.
     *
     * This is the entry point to access the GraphBuilder. If the instance has not
     * been created yet, it creates one in a thread-safe way.
     *
     * @return A reference to the single GraphBuilder instance.
     */
    static GraphBuilder &get_instance() {
        std::lock_guard lock(instance_mutex);
        if (!instance) {
            instance = std::unique_ptr<GraphBuilder>(new GraphBuilder());
        }
        return *instance;
    }

    /**
     * @brief Processes a network connection event and adds the corresponding nodes and edges to the graph.
     *
     * This method takes details of a network connection (e.g., source and destination IPs, ports, protocol)
     * and updates the traffic graph by adding or retrieving the involved nodes and creating an edge
     * representing the connection between them. It also updates the temporal features of the nodes.
     *
     * @param src_ip Source IP address of the connection.
     * @param dst_ip Destination IP address of the connection.
     * @param proto Protocol of the connection (e.g., "tcp", "udp").
     * @param timestamp Timestamp of the connection event.
     * @param src_port Source port of the connection.
     * @param dst_port Destination port of the connection.
     * @param method HTTP request method (if applicable). Defaults to "".
     * @param host HTTP host header (if applicable). Defaults to "".
     * @param uri HTTP request URI (if applicable). Defaults to "".
     * @param version HTTP version (if applicable). Defaults to "".
     * @param user_agent HTTP user agent string (if applicable). Defaults to "".
     * @param request_body_len Length of the HTTP request body (if applicable). Defaults to 0.
     * @param response_body_len Length of the HTTP response body (if applicable). Defaults to 0.
     * @param status_code HTTP status code (if applicable). Defaults to 0.
     * @param status_msg HTTP status message (if applicable). Defaults to "".
     * @param tags A vector of tags associated with the connection. Defaults to {}.
     * @param resp_fuids A vector of file unique identifiers from the responder. Defaults to {}.
     * @param resp_mime_types A vector of MIME types from the responder. Defaults to {}.
     */
    void add_connection(const std::string &src_ip, const std::string &dst_ip,
                        const std::string &proto, const std::string &timestamp,
                        int src_port, int dst_port, const std::string &method = "",
                        const std::string &host = "",
                        const std::string &uri = "",
                        const std::string &version = "",
                        const std::string &user_agent = "",
                        int request_body_len = 0,
                        int response_body_len = 0,
                        int status_code = 0,
                        const std::string &status_msg = "",
                        const std::vector<std::string> &tags = {},
                        const std::vector<std::string> &resp_fuids = {},
                        const std::vector<std::string> &resp_mime_types = {});

    /**
     * @brief Gets a reference to the underlying TrafficGraph object.
     *
     * This method provides access to the `TrafficGraph` object managed by the `GraphBuilder`.
     *
     * @return A reference to the TrafficGraph object.
     */
    TrafficGraph &get_graph();
};


#endif //GRAPHBUILDER_H
