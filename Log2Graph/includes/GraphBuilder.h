/**
 * @file GraphBuilder.h
 * @brief Header file for the GraphBuilder class, responsible for constructing the network traffic graph.
 *
 * This file defines the `GraphBuilder` class,
 * which form the data structures and logic for building a graph
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

#include "FeatureEncoder.h"
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
    /**
* @brief Feature encoder. Useful for passing data to GAT.
*/
    FeatureEncoder feature_encoder;
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
     * @brief Gets the update list for the graph.
     *
     * This is useful for periodic dumping, to just export incremental updates.
     *
     * @return
     */
    std::vector<GraphUpdate> get_last_updates() {
        return std::move(update_queue.popAll());
    }

/**
     * @brief Processes a network connection event and adds the corresponding nodes and edges to the graph,
     * including SSL and HTTP related information if available.
     *
     * This method takes details of a network connection (e.g., source and destination IPs, ports, protocol)
     * and updates the traffic graph by adding or retrieving the involved nodes and creating an edge
     * representing the connection between them. It also updates the temporal features of the nodes and
     * includes SSL-specific details such as version, cipher, and certificates, and HTTP-specific details
     * such as method, URI, and status code if provided.
     *
     * @param src_ip Source IP address of the connection.
     * @param dst_ip Destination IP address of the connection.
     * @param proto Protocol of the connection (e.g., "tcp", "udp").
     * @param service Service from a list of known services.
     * @param timestamp Timestamp of the connection event.
     * @param src_port Source port of the connection.
     * @param dst_port Destination port of the connection.
     * @param orig_bytes Number of bytes sent by the originator.
     * @param resp_bytes Number of bytes sent by the responder.
     * @param conn_state Connection state (e.g., "REJ", "SF").
     * @param local_orig Whether the originator is local.
     * @param local_resp Whether the responder is local.
     * @param history History of the connection (e.g., packet states).
     * @param orig_pkts Number of packets sent by the originator.
     * @param resp_pkts Number of packets sent by the responder.
     * @param orig_ip_bytes Number of IP bytes sent by the originator.
     * @param resp_ip_bytes Number of IP bytes sent by the responder.
     * @param version SSL/TLS version (if applicable).
     * @param cipher SSL/TLS cipher suite (if applicable).
     * @param curve Elliptic curve used for key exchange (if applicable).
     * @param server_name Server Name Indication (SNI) (if applicable).
     * @param resumed Whether the SSL/TLS session was resumed (if applicable).
     * @param last_alert The last SSL/TLS alert sent or received (if applicable).
     * @param next_protocol The next protocol negotiated via ALPN (if applicable).
     * @param established Whether the SSL/TLS connection was established (if applicable).
     * @param ssl_history SSL/TLS state history (if applicable).
     * @param cert_chain_fps Fingerprints of the server's certificate chain (if applicable).
     * @param client_cert_chain_fps Fingerprints of the client's certificate chain (if applicable).
     * @param sni_matches_cert Whether the SNI matched the presented certificate (if applicable).
     * @param http_method HTTP request method (if applicable).
     * @param http_host HTTP host header (if applicable).
     * @param http_uri HTTP request URI (if applicable).
     * @param http_referrer HTTP referrer header (if applicable).
     * @param http_version HTTP version (if applicable).
     * @param http_user_agent HTTP user-agent header (if applicable).
     * @param http_origin HTTP origin header (if applicable).
     * @param http_status_code HTTP response status code (if applicable).
     * @param http_username HTTP username (if applicable).
     */
    void add_connection(const std::string &src_ip, const std::string &dst_ip,
                        const std::string &proto, const std::string &service, const std::string &timestamp,
                        const int src_port, const int dst_port,
                        const int orig_bytes, const int resp_bytes,
                        const std::string &conn_state,
                        const bool local_orig, const bool local_resp,
                        const std::string &history,
                        const int orig_pkts, const int resp_pkts,
                        const int orig_ip_bytes, const int resp_ip_bytes,
                        const std::string &version = "",
                        const std::string &cipher = "",
                        const std::string &curve = "",
                        const std::string &server_name = "",
                        const bool resumed = false,
                        const std::string &last_alert = "",
                        const std::string &next_protocol = "",
                        const bool established = false,
                        const std::string &ssl_history = "",
                        const std::string &cert_chain_fps = "",
                        const std::string &client_cert_chain_fps = "",
                        const bool sni_matches_cert = false,
                        const std::string &http_method = "",
                        const std::string &http_host = "",
                        const std::string &http_uri = "",
                        const std::string &http_referrer = "",
                        const std::string &http_version = "",
                        const std::string &http_user_agent = "",
                        const std::string &http_origin = "",
                        const std::string &http_status_code = "",
                        const std::string &http_username = "");

    /**
     * @brief Gets a reference to the underlying TrafficGraph object.
     *
     * This method provides access to the `TrafficGraph` object managed by the `GraphBuilder`.
     *
     * @return A reference to the TrafficGraph object.
     */
    TrafficGraph &get_graph();

    /**
 * @brief Gets the dimension of the encoded feature vector
 * @return Size of feature vector produced by the encoder
 */
    size_t get_feature_dimension() const {
        // This should match your encoder's output size
        return feature_encoder.get_feature_dimension();
    }
};


#endif //GRAPHBUILDER_H
