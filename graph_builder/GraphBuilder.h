//
// Created by lu on 4/25/25.
//

#ifndef GRAPHBUILDER_H
#define GRAPHBUILDER_H
//
// Created by lu on 4/25/25.
//

#include "GraphBuilder.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

class GraphNode {
public:
    std::string id;
    std::string type; // "host", "service", "domain", etc.
    std::unordered_map<std::string, std::string> attributes;

    GraphNode(const std::string& id, const std::string& type)
        : id(id), type(type) {}
};

class GraphEdge {
public:
    std::string source;
    std::string target;
    std::string relationship;
    std::unordered_map<std::string, std::string> attributes;

    GraphEdge(const std::string& src, const std::string& tgt,
              const std::string& rel)
        : source(src), target(tgt), relationship(rel) {}
};

class TrafficGraph {
private:
    std::unordered_map<std::string, std::shared_ptr<GraphNode>> nodes;
    std::vector<std::shared_ptr<GraphEdge>> edges;
    mutable std::shared_mutex graph_mutex;

public:
    void add_node(const std::string& id, const std::string& type);

    void add_edge(const std::string& src, const std::string& tgt,
                 const std::string& rel, const std::unordered_map<std::string, std::string>& attrs = {});

    // Thread-safe graph access methods
    std::vector<std::shared_ptr<GraphNode>> get_nodes() const;

    // Analysis methods
    std::vector<std::string> find_connected_components() const;
    std::vector<std::string> detect_anomalies() const;
    // Add more analysis methods as needed
};

class GraphBuilder {
private:
    static std::unique_ptr<GraphBuilder> instance;
    static std::mutex instance_mutex;

    TrafficGraph graph;

    GraphBuilder() = default;

public:
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;

    static GraphBuilder& get_instance() {
        std::lock_guard lock(instance_mutex);
        if (!instance) {
            instance = std::unique_ptr<GraphBuilder>(new GraphBuilder());
        }
        return *instance;
    }

    void add_connection(const std::string& src_ip, const std::string& dst_ip,
                       const std::string& proto, const std::string& timestamp,
                       int src_port, int dst_port) ;
    TrafficGraph& get_graph();

};


#endif //GRAPHBUILDER_H
