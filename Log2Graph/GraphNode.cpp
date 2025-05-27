//
// Created by lu on 5/7/25.
//

#include "includes/GraphNode.h"
#include <algorithm>
#include <chrono>

GraphNode::GraphNode(const std::string &id, const std::string &type)
    : id(id), type(type) {
    temporal.monitoring_start = std::chrono::system_clock::now();
}

void GraphNode::update_connection_features(const std::string &protocol, bool is_outgoing) {
    // Update degree counts
    ++features.degree;
    if (is_outgoing) {
        ++features.out_degree;
    } else {
        ++features.in_degree;
    }

    // Update protocol distribution
    features.protocol_counts[protocol]++;

    // Update temporal features
    auto now = std::chrono::system_clock::now();
    {
        std::lock_guard<std::mutex> lock(temporal.recent_connections_mutex);
        temporal.recent_connections.push_back(now);
    }
    ++temporal.total_connections;
    {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);
        temporal.minute_window.push(now);
        temporal.hour_window.push(now);
    }

    // Clean up old connections
    cleanup_time_windows();
    // Update activity score (weighted moving average)
    double new_activity = 1.0; // Base weight for new connection
    features.activity_score = 0.9 * features.activity_score.load() + 0.1 * new_activity;

    // Periodically clean old connections
    if (temporal.recent_connections.size() % 10 == 0) {
        cleanup_old_connections();
    }
}

void GraphNode::cleanup_old_connections() {
    auto now = std::chrono::system_clock::now();
    auto cutoff = now - std::chrono::seconds(60); // Example: keep connections from the last 60 seconds

    std::lock_guard<std::mutex> lock(temporal.recent_connections_mutex);
    temporal.recent_connections.erase(
        std::remove_if(temporal.recent_connections.begin(), temporal.recent_connections.end(),
                         [&](const auto& time) { return time < cutoff; }),
        temporal.recent_connections.end());
}

double GraphNode::calculate_anomaly_score() const {
    // Placeholder for anomaly score calculation based on features
    // This is a simplified example and should be replaced with your actual logic
    double score = 0.0;
    score += features.degree.load() * 0.1;
    score += features.in_degree.load() * 0.05;
    score += features.out_degree.load() * 0.05;
    score += features.protocol_counts.size() * 0.02;
    score += features.activity_score.load() * 0.2;
    score += get_connections_last_minute() * 0.15;
    score += get_connections_last_hour() * 0.1;
    return score;
}

void GraphNode::cleanup_time_windows() {
    auto now = std::chrono::system_clock::now();
    auto minute_cutoff = now - std::chrono::seconds(60);
    auto hour_cutoff = now - std::chrono::hours(1);

    {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);
        while (!temporal.minute_window.empty() && temporal.minute_window.front() < minute_cutoff) {
            temporal.minute_window.pop();
        }
        while (!temporal.hour_window.empty() && temporal.hour_window.front() < hour_cutoff) {
            temporal.hour_window.pop();
        }
    }

    // Update atomic counters based on the window sizes (optional, can be done in get methods)
    temporal.connections_last_minute.store(temporal.minute_window.size());
    temporal.connections_last_hour.store(temporal.hour_window.size());
}

int GraphNode::get_connections_last_minute() const {
    return temporal.connections_last_minute.load();
}

int GraphNode::get_connections_last_hour() const {
    return temporal.connections_last_hour.load();
}