//
// Created by lu on 4/25/25.
//

#ifndef GRAPHBUILDER_H
#define GRAPHBUILDER_H
//
// Created by lu on 4/25/25.
//

#include <atomic>
#include <map>

#include "GraphBuilder.h"

#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

class GraphNode {
public:
    std::string id;
    std::string type; // "host", "service", "domain", etc.
    std::unordered_map<std::string, std::string> attributes;

    struct {
        std::atomic<int> degree{0};
        std::atomic<int> in_degree{0};
        std::atomic<int> out_degree{0};
        std::map<std::string, int> protocol_counts;
        std::atomic<double> activity_score{0.0};
    } features;

    // Temporal features
    struct {
        std::atomic<int> connections_last_minute{0};
        std::atomic<int> connections_last_hour{0};
        std::chrono::system_clock::time_point monitoring_start;
        std::atomic<int> total_connections{0};
        std::vector<std::chrono::system_clock::time_point> recent_connections;
        std::queue<std::chrono::system_clock::time_point> minute_window;
        std::queue<std::chrono::system_clock::time_point> hour_window;
        mutable std::mutex window_mutex; // Protects the queues
    } temporal;

    GraphNode(const std::string &id, const std::string &type)
        : id(id), type(type) {
        temporal.monitoring_start = std::chrono::system_clock::now();
    }

    void update_connection_features(const std::string &protocol, bool is_outgoing);

    void cleanup_old_connections();

    double calculate_anomaly_score() const;

    void cleanup_time_windows();

    int get_connections_last_minute() const;

    int get_connections_last_hour() const;
};

class GraphEdge {
public:
    std::string source;
    std::string target;
    std::string relationship;
    std::unordered_map<std::string, std::string> attributes;

    GraphEdge(const std::string &src, const std::string &tgt,
              const std::string &rel)
        : source(src), target(tgt), relationship(rel) {
    }
};

class TrafficGraph {
    friend class RealTimeAnomalyDetector;

private:
    std::unordered_map<std::string, std::shared_ptr<GraphNode> > nodes;
    std::vector<std::shared_ptr<GraphEdge> > edges;
    mutable std::shared_mutex graph_mutex;
    std::thread maintenance_thread;
    std::atomic<bool> running{true};

public:
    TrafficGraph() {
        maintenance_thread = std::thread([this]() {
            while (running) {
                clean_all_time_windows();
                std::this_thread::sleep_for(std::chrono::seconds(10));
            }
        });
    }

    ~TrafficGraph() {
        running = false;
        if (maintenance_thread.joinable()) {
            maintenance_thread.join();
        }
    }

    void clean_all_time_windows() {
        std::shared_lock lock(graph_mutex);
        for (auto &[id, node]: nodes) {
            node->cleanup_time_windows();
        }
    }

    void add_node(const std::string &id, const std::string &type);

    void add_edge(const std::string &src, const std::string &tgt,
                  const std::string &rel, const std::unordered_map<std::string, std::string> &attrs = {});

    GraphNode &get_or_create_node(const std::string &id, const std::string &type);

    // Thread-safe graph access methods
    std::vector<std::shared_ptr<GraphNode> > get_nodes() const;

    std::vector<std::shared_ptr<GraphEdge> > get_edges() const;

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
    GraphBuilder(const GraphBuilder &) = delete;

    GraphBuilder &operator=(const GraphBuilder &) = delete;

    static GraphBuilder &get_instance() {
        std::lock_guard lock(instance_mutex);
        if (!instance) {
            instance = std::unique_ptr<GraphBuilder>(new GraphBuilder());
        }
        return *instance;
    }

    void add_connection(const std::string &src_ip, const std::string &dst_ip,
                        const std::string &proto, const std::string &timestamp,
                        int src_port, int dst_port);

    TrafficGraph &get_graph();
};


#endif //GRAPHBUILDER_H
