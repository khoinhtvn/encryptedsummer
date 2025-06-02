//
// Created by lu on 4/28/25.
//

#include "includes/GraphExporter.h"
#include "includes/GraphNode.h" // Include GraphNode for accessing to_dot_string_encoded()
#include "includes/GraphEdge.h" // Include GraphEdge for accessing encoded features

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

void GraphExporter::export_incremental_update_encoded_async(std::vector<GraphUpdate> updates,
                                                            const std::string &output_file) {
    std::thread t(&GraphExporter::export_incremental_update_worker, this, updates, output_file);
    t.detach(); // Detach the thread so it runs independently
}

void GraphExporter::export_full_graph_worker(const TrafficGraph &graph,
                                             const std::string &output_file) {
    if (!graph.is_empty()) {
        export_to_dot_encoded(graph, output_file);
    } else {
        std::cout << "Empty graph!" << std::endl;
    }
}

void GraphExporter::export_incremental_update_worker(std::vector<GraphUpdate> updates,
                                                     const std::string &output_file_base) {
    if (!updates.empty()) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm now_gmt;
        gmtime_r(&now_c, &now_gmt);
        std::stringstream ss;
        ss << output_file_base.substr(0, output_file_base.find_last_of('.'))
           << "_update_" << std::put_time(&now_gmt, "%Y%m%d_%H%M%S_UTC") << ".dot";
        std::string output_file = ss.str();

        std::ofstream ofs(output_file);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open output file: " + output_file);
        }

        // Write DOT format header
        ofs << "digraph NWTraffic_update {\n";
        for (const auto &update : updates) {
            switch (update.type) {
                case GraphUpdate::Type::NODE_CREATE:
                case GraphUpdate::Type::NODE_UPDATE:
                    if (auto node = update.node.lock()) {
                        write_node_encoded_to_file(node, ofs);
                    }
                    break;
                case GraphUpdate::Type::EDGE_CREATE:
                    if (auto edge = update.edge.lock()) {
                        write_edge_encoded_to_file(edge, ofs);
                    }
                    break;
                default:
                    std::cerr << "Warning: Unknown GraphUpdate type encountered." << std::endl;
                    break;
            }
        }
        ofs << "}\n";
        ofs.close();
        std::cout << "File " << output_file << " written" << std::endl;
    } else {
        std::cout << "Empty updates vector!" << std::endl;
    }
}

void GraphExporter::write_node_encoded_to_file(const std::shared_ptr<GraphNode> &node, std::ofstream &ofstream) {
    if (node) {
        ofstream << node->to_dot_string_encoded();
    }
}

void GraphExporter::write_edge_encoded_to_file(const std::shared_ptr<GraphEdge> &edge, std::ofstream &ofstream) {
    if (edge) {
        ofstream << edge->to_dot_string_encoded();
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
    for (const auto &edge : graph.get_edges()) {
        write_edge_encoded_to_file(edge, dot_file);
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