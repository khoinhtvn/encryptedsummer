//
// Created by lu on 4/25/25.
//

#include "GraphBuilder.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

std::unique_ptr<GraphBuilder> GraphBuilder::instance = nullptr;
std::mutex GraphBuilder::instance_mutex;


    void TrafficGraph::add_node(const std::string& id, const std::string& type) {
        std::unique_lock lock(graph_mutex);
        if (nodes.find(id) == nodes.end()) {
            nodes[id] = std::make_shared<GraphNode>(id, type);
        }
    }

    void TrafficGraph::add_edge(const std::string& src, const std::string& tgt,
                 const std::string& rel, const std::unordered_map<std::string, std::string>& attrs) {
        std::unique_lock lock(graph_mutex);
        auto edge = std::make_shared<GraphEdge>(src, tgt, rel);
        edge->attributes = attrs;
        edges.push_back(edge);
    }

    // Thread-safe graph access methods
    std::vector<std::shared_ptr<GraphNode>> TrafficGraph::get_nodes() const {
        std::shared_lock lock(graph_mutex);
        std::vector<std::shared_ptr<GraphNode>> result;
        for (const auto& pair : nodes) {
            result.push_back(pair.second);
        }
        return result;
    }

std::vector<std::shared_ptr<GraphEdge>> TrafficGraph::get_edges() const {
        return edges;
    }

    void GraphBuilder::add_connection(const std::string& src_ip, const std::string& dst_ip,
                       const std::string& proto, const std::string& timestamp,
                       int src_port, int dst_port) {
        // Add nodes
        graph.add_node(src_ip, "host");
        graph.add_node(dst_ip, "host");

        // Add edge attributes
        std::unordered_map<std::string, std::string> attrs = {
            {"protocol", proto},
            {"timestamp", timestamp},
            {"src_port", std::to_string(src_port)},
            {"dst_port", std::to_string(dst_port)}
        };

        // Add edge
        graph.add_edge(src_ip, dst_ip, proto + "_connection", attrs);
    }

    TrafficGraph& GraphBuilder::get_graph() { return graph; }

