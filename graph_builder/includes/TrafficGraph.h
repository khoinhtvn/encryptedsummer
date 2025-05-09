//
// Created by lu on 5/7/25.
//

#ifndef TRAFFICGRAPH_H
#define TRAFFICGRAPH_H
#include <memory>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "GraphEdge.h"
#include "GraphNode.h"
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
    /**
     * @brief Atomic integer counter to keep track of the changes since the last incremental export.
     */
    std::atomic<size_t> update_counter{0};

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
     * Creates a new `GraphEdge` object, adds it to the list of edges and returns a weak reference.
     *
     * @param src The identifier of the source node.
     * @param tgt The identifier of the target node.
     * @param rel The type of relationship between the nodes.
     * @param attrs Optional attributes to associate with the edge. Defaults to an empty map.
     * @param attrs Optional encoded featrures to associate with the edge. Defaults to an empty vector.
     * @return A weak reference to the GraphEdge. Useful for incremental updates.
     */
    std::weak_ptr<GraphEdge> add_edge(const std::string &src, const std::string &tgt,
                                      const std::string &rel,
                                      const std::unordered_map<std::string, std::string> &attrs = {}, std::vector<float> encoded_features = {});

    /**
     * @brief Retrieves an existing node or creates a new one if it doesn't exist.
     *
     * This method first tries to find a node with the given ID. If found, it returns a
     * reference to it. If not found, it creates a new node with the given ID and type,
     * adds it to the graph, and then returns a reference to the newly created node.
     *
     * @param id The unique identifier of the node.
     * @param type The type of the node.
     * @return A pair containing a reference to the GraphNode and a boolean indicating if the node was freshly created.
     */
    std::pair<GraphNode &, bool> get_or_create_node(const std::string &id, const std::string &type);

    /**
     * @brief Retrieves a weak reference to an existing node. Used to create updatre queue.
     *
     * @param id The unique identifier of the node.
     * @return A reference to the GraphNode.
     */
    std::weak_ptr<GraphNode> get_node_reference(const std::string &id);

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
#endif //TRAFFICGRAPH_H
