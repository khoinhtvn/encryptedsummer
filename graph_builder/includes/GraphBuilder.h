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

#include <memory>
#include <mutex>
#include <vector>

#include "GraphUpdateQueue.h"
#include "TrafficGraph.h"


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
     * @brief Default Constructor. Private to enforce signeton pattern.
     */
    GraphBuilder() = default;

    /**
 * @brief Queue of incremental updates. Useful for passing just new graph features when needed.
 */
    GraphUpdateQueue update_queue;
    std::atomic<bool> save_pending{false};

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
