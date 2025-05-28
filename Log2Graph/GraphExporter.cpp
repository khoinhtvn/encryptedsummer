//
// Created by lu on 4/28/25.
//

#include "includes/GraphExporter.h"
#include "includes/GraphNode.h" // Include GraphNode for accessing features

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>

GraphExporter::GraphExporter() {}

GraphExporter::~GraphExporter() {}

void GraphExporter::export_full_graph_human_readable_async(const TrafficGraph &graph,
                                                             const std::string &output_file,
                                                             bool open_image, bool export_cond) {
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
        export_to_dot(graph, output_file + ".dot");
    } else if (graph.is_empty()) {
        std::cout << "Empty graph!" << std::endl;
    }
}

void GraphExporter::export_incremental_update_worker(std::vector<GraphUpdate> updates,
                                                         const std::string &output_file) {
    if (!updates.empty()) {
        std::ofstream ofs(output_file);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open output file.");
        }

        // Write DOT format header
        ofs << "digraph NWTraffic_update {\n";
        for (const auto &update: updates) {
            switch (update.type) {
                case GraphUpdate::Type::NODE_CREATE:
                    if (auto node = update.node.lock()) {
                        ofs << node->to_dot_string();
                    }
                    break;
                case GraphUpdate::Type::EDGE_CREATE:
                    if (auto edge = update.edge.lock()) {
                        ofs << "  \"" << GraphEdge::escape_dot_string(edge->source) << "\" -> \"" << GraphEdge::escape_dot_string(edge->target) << "\" [";
                        for (size_t i = 0; i < edge->encoded_features.size(); ++i) {
                            if (i != 0) ofs << ",";
                            ofs << FeatureEncoder::get_feature_name(i) << "="
                                << edge->encoded_features[i];
                        }
                        ofs << "  ];\n";
                    }
                    break;
                case GraphUpdate::Type::NODE_UPDATE:
                    if (auto node = update.node.lock()) {
                        ofs << node->to_dot_string();
                    }
                    break;
                default:
                    throw std::runtime_error("Unknown GraphUpdate type.");
            }
        }
        ofs << "}\n";
        ofs.close();
        std::cout << "File " << output_file << " written" << std::endl;
    } else {
        std::cout << "Empty updates vector!" << std::endl;
    }
}

void GraphExporter::export_to_dot(const TrafficGraph &graph, const std::string &filename) {
    std::ofstream dot_file(filename);
    dot_file << "digraph ZeekTraffic {\n";

    for (const auto &node: graph.get_nodes()) {
        dot_file << node->to_dot_string();
    }

    for (const auto &edge: graph.get_edges()) {
        dot_file << edge->to_dot_string();
    }

    dot_file << "}\n";
    dot_file.close();
    std::cout << "File " << filename << " written" << std::endl;
}