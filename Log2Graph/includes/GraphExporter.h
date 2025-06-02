#ifndef GRAPHEXPORTER_H
#define GRAPHEXPORTER_H

#include <vector>   // For using std::vector
#include <string>   // For using std::string
#include <memory>   // For using std::shared_ptr
#include <fstream>  // For file stream operations (std::ofstream)

#include "TrafficGraph.h"    // For accessing the TrafficGraph data structure
#include "GraphNode.h"       // For representing nodes in the graph
#include "AggregatedGraphEdge.h" // For representing aggregated edges in the graph

/**
 * @brief Class responsible for exporting the network traffic graph to various formats,
 * currently focusing on the DOT language for graph visualization.
 */
class GraphExporter {
public:
    /**
     * @brief Default constructor for the GraphExporter.
     */
    GraphExporter();

    /**
     * @brief Virtual destructor for proper cleanup of resources.
     */
    ~GraphExporter();

    /**
     * @brief Exports the entire traffic graph (nodes and aggregated edges) to a DOT file asynchronously.
     *
     * The export operation is performed in a separate thread to avoid blocking the main thread.
     * The output includes encoded features of nodes and edges.
     *
     * @param graph The TrafficGraph object to export.
     * @param output_file The path to the file where the DOT representation will be written.
     */
    void export_full_graph_encoded_async(const TrafficGraph &graph, const std::string &output_file);


private:
    /**
     * @brief Worker function executed in a separate thread to export the full graph to a DOT file.
     *
     * @param graph The TrafficGraph object to export (passed by reference).
     * @param output_file The path to the output DOT file.
     */
    void export_full_graph_worker(const TrafficGraph &graph, const std::string &output_file);


    /**
     * @brief Exports the entire traffic graph (nodes and aggregated edges) to a DOT file.
     *
     * This is the synchronous version of the full graph export.
     *
     * @param graph The TrafficGraph object to export.
     * @param filename The path to the output DOT file.
     */
    void export_to_dot_encoded(const TrafficGraph &graph, const std::string &filename);

    /**
     * @brief Writes the DOT representation of a single GraphNode (including encoded features) to the output file stream.
     *
     * @param node A shared pointer to the GraphNode to write.
     * @param ofstream A reference to the output file stream.
     */
    void write_node_encoded_to_file(const std::shared_ptr<GraphNode> &node, std::ofstream &ofstream);

    /**
     * @brief Writes the DOT representation of a single GraphEdge (including encoded features) to the output file stream.
     *
     * This is for legacy individual edges.
     *
     * @param edge A shared pointer to the GraphEdge to write.
     * @param ofstream A reference to the output file stream.
     */
    void write_edge_encoded_to_file(const std::shared_ptr<AggregatedGraphEdge> &edge, std::ofstream &ofstream);

    /**
     * @brief Writes the DOT representation of a single AggregatedGraphEdge (including aggregated information and potentially features) to the output file stream.
     *
     * @param edge A shared pointer to the AggregatedGraphEdge to write.
     * @param ofstream A reference to the output file stream.
     */
    void write_aggregated_edge_encoded_to_file(const std::shared_ptr<AggregatedGraphEdge> &edge, std::ofstream &ofstream);

    /**
     * @brief Escapes special characters in a string so it can be safely used within a DOT language label.
     *
     * @param str The input string to escape.
     * @return The escaped string.
     */
    std::string escape_dot_string(const std::string &str);
};

#endif // GRAPHEXPORTER_H