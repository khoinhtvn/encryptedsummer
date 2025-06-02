//
// Created by lu on 4/28/25.
//

#ifndef GRAPH_EXPORTER_H
#define GRAPH_EXPORTER_H

#include "GraphBuilder.h"
#include <string>
#include <memory>
#include <future>
#include <fstream> // For file output

/**
 * @brief Class responsible for exporting the network traffic graph with encoded features.
 *
 * The `GraphExporter` class takes a `TrafficGraph` object and exports its nodes and
 * edges with their encoded features to a DOT file.
 */
class GraphExporter {
public:
    /**
     * @brief Constructor for the GraphExporter.
     */
    GraphExporter();

    /**
     * @brief Destructor for the GraphExporter.
     */
    ~GraphExporter();

    /**
     * @brief Exports the full traffic graph with encoded node and edge features
     * to a DOT file in a separate thread.
     *
     * This method takes a `TrafficGraph`, retrieves the encoded string representation
     * of each node and edge, and saves it to the specified output file
     * (defaulting to "full_graph_encoded.dot") in a separate thread.
     *
     * @param graph The `TrafficGraph` object to export.
     * @param output_file The name of the output DOT file (default: "full_graph_encoded.dot").
     */
    void export_full_graph_encoded_async(const TrafficGraph &graph,
                                          const std::string &output_file = "full_graph_encoded.dot");

    /**
     * @brief Exports the encoded incremental updates of the graph to a DOT file
     * in a separate thread.
     *
     * This method retrieves the latest incremental updates from the GraphBuilder,
     * and saves the encoded representation of the created/updated nodes and edges
     * to a specified DOT file in a separate thread. The filename includes a
     * timestamp (UTC) to ensure uniqueness for each incremental update export.
     *
     * @param updates A vector of `GraphUpdate` objects representing the incremental changes
     * to the graph. This is typically obtained from the GraphBuilder.
     * @param output_file The full path and filename for the output DOT file.
     * It defaults to "update_encoded.dot". The function call
     * constructs a more specific path including the base
     * `export_path`, a separator, the prefix "nw_graph_update_",
     * a UTC timestamp, and the ".dot" extension.
     */
    void export_incremental_update_encoded_async(std::vector<GraphUpdate> updates,
                                                  const std::string &output_file = "update_encoded.dot");

private:
    /**
     * @brief Worker function to perform the full graph export with encoded features.
     *
     * @param graph The `TrafficGraph` object to export.
     * @param output_file The name of the output DOT file.
     */
    void export_full_graph_worker(const TrafficGraph &graph,
                                   const std::string &output_file);

    /**
     * @brief Worker function to export incremental updates with encoded features
     * to a DOT file.
     *
     * @param updates A vector of `GraphUpdate` objects representing the incremental changes.
     * @param output_file The full path and filename for the output DOT file.
     */
    void export_incremental_update_worker(std::vector<GraphUpdate> updates,
                                            const std::string &output_file);

    /**
     * @brief Writes the encoded string representation of a node to the output stream.
     *
     * @param node The shared pointer to the GraphNode to write.
     * @param ofstream The output file stream.
     */
    void write_node_encoded_to_file(const std::shared_ptr<GraphNode> &node, std::ofstream &ofstream);

    /**
     * @brief Writes the encoded string representation of an edge to the output stream.
     *
     * @param edge The shared pointer to the GraphEdge to write.
     * @param ofstream The output file stream.
     */
    void write_edge_encoded_to_file(const std::shared_ptr<GraphEdge> &edge, std::ofstream &ofstream);

    /**
     * @brief Exports the `TrafficGraph` structure to a DOT file with encoded features.
     *
     * @param graph The `TrafficGraph` to export.
     * @param filename The name of the output DOT file.
     */
    void export_to_dot_encoded(const TrafficGraph &graph, const std::string &filename);

    std::string escape_dot_string(const std::string &str);
};

#endif // GRAPH_EXPORTER_H