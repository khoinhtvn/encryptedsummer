//
// Created by lu on 4/28/25.
//

#ifndef GRAPH_EXPORTER_H
#define GRAPH_EXPORTER_H

#include "GraphBuilder.h"
#include <graphviz/gvc.h>
#include <string>
#include <memory>
#include <future>

/**
 * @brief Class responsible for visualizing the network traffic graph using Graphviz.
 *
 * The `GraphExporter` class takes a `TrafficGraph` object and renders it as a visual
 * representation, typically saving it as a PNG image. It utilizes the Graphviz library
 * for graph layout and rendering.
 */
class GraphExporter {
public:
    /**
     * @brief Constructor for the GraphExporter.
     *
     * Initializes the Graphviz context (`GVC_t`).
     */
    GraphExporter();

    /**
     * @brief Destructor for the GraphExporter.
     *
     * Releases the Graphviz context to free allocated resources.
     */
    ~GraphExporter();

    /**
     * @brief Visualizes the given traffic graph and saves it to a file in a separate thread.
     *
     * This method takes a `TrafficGraph`, generates a visual representation using
     * Graphviz, and saves it to the specified output file (defaulting to "graph.png")
     * in a separate thread, allowing the program to continue execution.
     * It can also optionally open the generated image and control whether the export
     * should proceed.
     *
     * @param graph The `TrafficGraph` object to visualize.
     * @param output_file The name of the output image file (default: "graph.png").
     * @param open_image A boolean indicating whether to attempt to open the generated image (default: true).
     * @param export_cond A boolean controlling whether the export process should proceed (default: true).
     */
    void export_full_graph_human_readable_async(const TrafficGraph &graph,
                                               const std::string &output_file = "graph.png",
                                               bool open_image = true, bool export_cond = true);

    /**
     * @brief Exports the encoded incremental updates of the graph to a DOT file in a separate thread.
     *
     * This method retrieves the latest incremental updates from the GraphBuilder,
     * encodes them, and then saves the encoded representation to a specified
     * DOT file in a separate thread. The filename includes a timestamp (UTC) to ensure uniqueness
     * for each incremental update export.
     *
     * @param updates A vector of `GraphUpdate` objects representing the incremental changes
     * to the graph. This is typically obtained from the GraphBuilder.
     * @param output_file The full path and filename for the output DOT file.
     * It defaults to "update.dot" if no path is provided. The
     * function call constructs a more specific path including
     * the base `export_path`, a separator, the prefix
     * "nw_graph_encoded_", a UTC timestamp, and the ".dot" extension.
     */
    void export_incremental_update_encoded_async(std::vector<GraphUpdate> updates,
                                                 const std::string &output_file = "update.dot");

private:
    /**
     * @brief Graphviz context object.
     *
     * This context is required for interacting with the Graphviz library.
     */
    GVC_t *gvc;

    /**
     * @brief Worker function to perform the Graphviz export.
     *
     * @param graph The `TrafficGraph` object to visualize.
     * @param output_file The name of the output image file.
     * @param open_image A boolean indicating whether to attempt to open the generated image.
     * @param export_cond A boolean controlling whether the export process should proceed.
     */
    void export_full_graph_worker(const TrafficGraph &graph,
                                  const std::string &output_file,
                                  bool open_image, bool export_cond);

    /**
     * @brief Worker function to export incremental updates to a DOT file.
     *
     * @param updates A vector of `GraphUpdate` objects representing the incremental changes.
     * @param output_file The full path and filename for the output DOT file.
     */
    void export_incremental_update_worker(std::vector<GraphUpdate> updates,
                                           const std::string &output_file);

    /**
     * @brief Adds nodes from the `TrafficGraph` to the Graphviz graph.
     *
     * Iterates through the nodes in the `TrafficGraph` and creates corresponding
     * nodes in the Graphviz graph (`Agraph_t`).
     *
     * @param graph The Graphviz graph to add nodes to.
     * @param traffic_graph The `TrafficGraph` containing the nodes to visualize.
     */
    void add_nodes(Agraph_t *graph, const TrafficGraph &traffic_graph);

    /**
     * @brief Adds edges from the `TrafficGraph` to the Graphviz graph.
     *
     * Iterates through the edges in the `TrafficGraph` and creates corresponding
     * edges in the Graphviz graph (`Agraph_t`), connecting the appropriate nodes.
     *
     * @param graph The Graphviz graph to add edges to.
     * @param traffic_graph The `TrafficGraph` containing the edges to visualize.
     */
    void add_edges(Agraph_t *graph, const TrafficGraph &traffic_graph);

    /**
     * @brief Applies default visual styles to the Graphviz graph.
     *
     * Sets basic attributes for the graph, such as layout engine, node styles,
     * and edge styles, to provide a consistent and readable visualization.
     *
     * @param graph The Graphviz graph to apply styles to.
     */
    void apply_default_styles(Agraph_t *graph);

    /**
     * @brief Generates a valid Graphviz node ID from the original node ID.
     *
     * Ensures that node IDs are compatible with Graphviz's requirements,
     * potentially escaping or modifying characters that could cause issues.
     *
     * @param original_id The original node identifier from the `TrafficGraph`.
     * @return A string representing a valid Graphviz node ID.
     */
    std::string generate_node_id(const std::string &original_id);

    void write_node_to_file(const std::shared_ptr<GraphNode> & node, std::ofstream & ofstream);

    void write_edge_to_file(const std::shared_ptr<GraphEdge> & edge, std::ofstream & ofstream);

    /**
     * @brief Exports the `TrafficGraph` structure to a DOT file.
     *
     * This method can be used for debugging or if a DOT representation of the
     * graph is needed. The DOT file can then be processed by Graphviz separately.
     *
     * @param graph The `TrafficGraph` to export.
     * @param filename The name of the output DOT file.
     */
    void export_to_dot(const TrafficGraph &graph, const std::string &filename);

    std::string escape_dot_string(const std::string &str);
};

#endif // GRAPH_EXPORTER_H