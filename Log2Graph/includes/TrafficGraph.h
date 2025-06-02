#include <memory>        // For using std::shared_ptr
#include <mutex>         // For using std::mutex to ensure thread safety
#include <chrono>        // For time-related functionalities (e.g., std::chrono::seconds)
#include <string>        // Include for std::string to represent node and edge identifiers
#include <unordered_map> // Include for std::unordered_map to efficiently store nodes by ID
#include <vector>        // Include for std::vector to store lists of nodes and aggregated edges

#include "GraphNode.h"         // Include the definition of the GraphNode class
#include "AggregatedGraphEdge.h" // Include the definition of the AggregatedGraphEdge class

#ifndef TRAFFICGRAPH_H
#define TRAFFICGRAPH_H

/**
 * @brief Class representing the network traffic graph.
 *
 * This class manages nodes (representing network entities like IP addresses) and
 * aggregated edges (representing summarized traffic flows between nodes).
 * It provides methods to add, retrieve, and manage these graph elements.
 */
class TrafficGraph {
public:
    /**
     * @brief Default constructor for the TrafficGraph.
     */
    TrafficGraph();

    /**
     * @brief Virtual destructor for proper cleanup of resources.
     */
    ~TrafficGraph();

    /**
     * @brief Retrieves an existing node with the given ID and type, or creates a new one if it doesn't exist.
     *
     * This method ensures that if a node with the specified ID already exists, it is returned.
     * If not, a new GraphNode with the given ID and type is created, added to the graph, and then returned.
     *
     * @param id The unique identifier of the node (e.g., an IP address).
     * @param type The type of the node (e.g., "internal", "external").
     * @return A shared pointer to the retrieved or newly created GraphNode.
     */
    std::shared_ptr<GraphNode> get_or_create_node(const std::string &id, const std::string &type);

    /**
     * @brief Adds an already created GraphNode to the graph.
     *
     * @param node A shared pointer to the GraphNode to add.
     */
    void add_node(std::shared_ptr<GraphNode> node);

    /**
     * @brief Adds an aggregated edge to the graph.
     *
     * Aggregated edges summarize multiple individual connections between two nodes.
     *
     * @param edge The AggregatedGraphEdge object to add.
     */
    void add_aggregated_edge(const AggregatedGraphEdge &edge);

    /**
     * @brief Retrieves a node with the given ID from the graph.
     *
     * @param id The unique identifier of the node to retrieve.
     * @return A shared pointer to the GraphNode if found, otherwise a null pointer or similar indication.
     */
    std::shared_ptr<GraphNode> get_node(const std::string &id) const;

    /**
     * @brief Returns a vector containing shared pointers to all nodes in the graph.
     *
     * @return A vector of std::shared_ptr<GraphNode>.
     */
    std::vector<std::shared_ptr<GraphNode>> get_nodes() const;

    /**
     * @brief Returns a vector containing all aggregated edges in the graph.
     *
     * @return A vector of AggregatedGraphEdge objects.
     */
    std::vector<AggregatedGraphEdge> get_aggregated_edges() const;

    /**
     * @brief Returns the total number of nodes in the graph.
     *
     * @return The number of nodes (size of the nodes_ map).
     */
    size_t get_node_count() const;

    /**
     * @brief Returns the total number of aggregated edges in the graph.
     *
     * @return The number of aggregated edges (size of the aggregated_edges_ vector).
     */
    size_t get_aggregated_edge_count() const;

    /**
     * @brief Checks if the graph contains any nodes or aggregated edges.
     *
     * @return True if the graph is empty (no nodes and no aggregated edges), false otherwise.
     */
    bool is_empty() const;

    /**
     * @brief Aggregates "old" individual edges based on a time threshold.
     *
     * This method iterates through the individual edges and aggregates those that haven't been updated
     * within the specified time frame into aggregated edges. After aggregation, the old individual
     * edges are likely removed.
     *
     * @param age_threshold The maximum age (duration) for an individual edge before it's considered "old" for aggregation.
     */
    void aggregate_old_edges(std::chrono::seconds age_threshold);

private:
    /**
     * @brief Internal storage for the graph nodes, using a map for efficient lookup by ID.
     *
     * The keys are node IDs (strings), and the values are shared pointers to the corresponding GraphNode objects.
     */
    std::unordered_map<std::string, std::shared_ptr<GraphNode>> nodes_;

    /**
     * @brief Internal storage for the aggregated edges, using a vector to hold all aggregated connections.
     */
    std::vector<AggregatedGraphEdge> aggregated_edges_;

    /**
     * @brief Mutex to protect access to the graph data structures (nodes_ and aggregated_edges_)
     * in a multithreaded environment. The mutex is mutable because const methods might need to lock it.
     */
    mutable std::mutex graph_mutex_;

    /**
     * @brief Recalculates the degree (number of connections) for all nodes in the graph.
     *
     * This method is likely called after adding or removing edges to ensure the node degree information is up-to-date.
     */
    void recalculate_node_degrees();
};

#endif // TRAFFICGRAPH_H