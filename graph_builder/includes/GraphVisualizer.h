/**
 * @file GraphVisualizer.h
 * @brief Header file for the GraphVisualizer class, responsible for visualizing the network traffic graph.
 *
 * This file defines the `GraphVisualizer` class, which uses the Graphviz library
 * to generate visual representations of the `TrafficGraph`.
 */

// Created by lu on 4/28/25.
//

#ifndef GRAPH_VISUALIZER_H
#define GRAPH_VISUALIZER_H

#include "GraphBuilder.h"
#include <graphviz/gvc.h>
#include <string>
#include <memory>

#include "GraphBuilder.h" // Self-include is unusual and likely unintentional. Remove in actual code.

/**
 * @brief Class responsible for visualizing the network traffic graph using Graphviz.
 *
 * The `GraphVisualizer` class takes a `TrafficGraph` object and renders it as a visual
 * representation, typically saving it as a PNG image. It utilizes the Graphviz library
 * for graph layout and rendering.
 */
class GraphVisualizer {
public:
    /**
     * @brief Constructor for the GraphVisualizer.
     *
     * Initializes the Graphviz context (`GVC_t`).
     */
    GraphVisualizer();

    /**
     * @brief Destructor for the GraphVisualizer.
     *
     * Releases the Graphviz context to free allocated resources.
     */
    ~GraphVisualizer();

    /**
     * @brief Visualizes the given traffic graph and saves it to a file.
     *
     * This method takes a `TrafficGraph`, generates a visual representation using
     * Graphviz, and saves it to the specified output file (defaulting to "graph.png").
     * It can also optionally open the generated image and control whether the export
     * should proceed.
     *
     * @param graph The `TrafficGraph` object to visualize.
     * @param output_file The name of the output image file (default: "graph.png").
     * @param open_image A boolean indicating whether to attempt to open the generated image (default: true).
     * @param export_cond A boolean controlling whether the export process should proceed (default: true).
     */
    void visualize(const TrafficGraph& graph,
                   const std::string& output_file = "graph.png",
                   bool open_image = true,  bool export_cond = true);

private:
    /**
     * @brief Graphviz context object.
     *
     * This context is required for interacting with the Graphviz library.
     */
    GVC_t* gvc;

    /**
     * @brief Adds nodes from the `TrafficGraph` to the Graphviz graph.
     *
     * Iterates through the nodes in the `TrafficGraph` and creates corresponding
     * nodes in the Graphviz graph (`Agraph_t`).
     *
     * @param graph The Graphviz graph to add nodes to.
     * @param traffic_graph The `TrafficGraph` containing the nodes to visualize.
     */
    void add_nodes(Agraph_t* graph, const TrafficGraph& traffic_graph);

    /**
     * @brief Adds edges from the `TrafficGraph` to the Graphviz graph.
     *
     * Iterates through the edges in the `TrafficGraph` and creates corresponding
     * edges in the Graphviz graph (`Agraph_t`), connecting the appropriate nodes.
     *
     * @param graph The Graphviz graph to add edges to.
     * @param traffic_graph The `TrafficGraph` containing the edges to visualize.
     */
    void add_edges(Agraph_t* graph, const TrafficGraph& traffic_graph);

    /**
     * @brief Applies default visual styles to the Graphviz graph.
     *
     * Sets basic attributes for the graph, such as layout engine, node styles,
     * and edge styles, to provide a consistent and readable visualization.
     *
     * @param graph The Graphviz graph to apply styles to.
     */
    void apply_default_styles(Agraph_t* graph);

    /**
     * @brief Generates a valid Graphviz node ID from the original node ID.
     *
     * Ensures that node IDs are compatible with Graphviz's requirements,
     * potentially escaping or modifying characters that could cause issues.
     *
     * @param original_id The original node identifier from the `TrafficGraph`.
     * @return A string representing a valid Graphviz node ID.
     */
    std::string generate_node_id(const std::string& original_id);

    /**
     * @brief Exports the `TrafficGraph` structure to a DOT file.
     *
     * This method can be used for debugging or if a DOT representation of the
     * graph is needed. The DOT file can then be processed by Graphviz separately.
     *
     * @param graph The `TrafficGraph` to export.
     * @param filename The name of the output DOT file.
     */
    void export_to_dot(const TrafficGraph& graph, const std::string& filename);

    std::string escape_dot_string(const std::string &str);
};

#endif // GRAPH_VISUALIZER_H