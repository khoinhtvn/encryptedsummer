//
// Created by lu on 4/28/25.
//

#include "includes/GraphExporter.h"
#include "includes/GraphNode.h" // Include GraphNode for accessing to_dot_string_encoded()
#include "includes/AggregatedGraphEdge.h" // Include AggregatedGraphEdge for accessing encoded features

#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <sstream>
#include <ctime>

GraphExporter::GraphExporter() {}

GraphExporter::~GraphExporter() {}

void GraphExporter::export_full_graph_encoded_async(const TrafficGraph &graph,
                                                 const std::string &output_file) {
    std::thread t(&GraphExporter::export_full_graph_worker, this, std::ref(graph), output_file);
    t.detach(); // Detach the thread so it runs independently
}
/*
void GraphExporter::export_incremental_update_encoded_async(std::vector<GraphUpdate> updates,
                                                          const std::string &output_file) {
    std::thread t(&GraphExporter::export_incremental_update_worker, this, updates, output_file);
    t.detach(); // Detach the thread so it runs independently
}
*/
void GraphExporter::export_full_graph_worker(const TrafficGraph &graph,
                                           const std::string &output_file) {
    if (!graph.is_empty()) {
        export_to_dot_encoded(graph, output_file);
    } else {
        std::cout << "Empty graph!" << std::endl;
    }
}

void GraphExporter::write_node_encoded_to_file(const std::shared_ptr<GraphNode> &node, std::ofstream &ofstream) {
    if (node) {
        ofstream << node->to_dot_string_encoded();
    }
}

void GraphExporter::write_aggregated_edge_encoded_to_file(const std::shared_ptr<AggregatedGraphEdge> &edge, std::ofstream &ofstream) {
    if (edge) {
        ofstream << edge->to_dot_string_aggregated();
    }
}

void GraphExporter::export_to_dot_encoded(const TrafficGraph &graph, const std::string &filename) {
    std::ofstream dot_file(filename);
    if (!dot_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    dot_file << "digraph ZeekTraffic {\n";

    for (const auto &node_ptr : graph.get_nodes()) {
        if (node_ptr) {
            write_node_encoded_to_file(node_ptr, dot_file);
        }
    }
    for (const auto &aggregated_edge : graph.get_aggregated_edges()) {
        write_aggregated_edge_encoded_to_file(std::make_shared<AggregatedGraphEdge>(aggregated_edge), dot_file);
    }

    dot_file << "}\n";
    dot_file.close();
    std::cout << "File " << filename << " written" << std::endl;
}

std::string GraphExporter::escape_dot_string(const std::string &str) {
    std::string result = "";
    for (char c : str) {
        if (c == '"') {
            result += "\\\"";
        } else if (c == '\\') {
            result += "\\\\";
        } else {
            result += c;
        }
    }
    return result;
}