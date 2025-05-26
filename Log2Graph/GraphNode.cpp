//
// Created by lu on 5/7/25.
//
#include <algorithm>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "includes/GraphNode.h"

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
    temporal.recent_connections.push_back(now);
    ++temporal.total_connections; {
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

    // Remove connections older than 1 hour
    auto hour_ago = now - std::chrono::hours(1);
    temporal.recent_connections.erase(
        std::remove_if(temporal.recent_connections.begin(),
                       temporal.recent_connections.end(),
                       [hour_ago](const auto &time) {
                           return time < hour_ago;
                       }),
        temporal.recent_connections.end()
    );
}

double GraphNode::calculate_anomaly_score() const {
    double last_min = get_connections_last_minute();
    double last_hour = get_connections_last_hour();
    double monitoring_minutes = std::chrono::duration<double>(
                                    std::chrono::system_clock::now() - temporal.monitoring_start).count() / 60.0;

    // Normalize hour count to per-minute rate
    double hour_rate = last_hour / std::min(60.0, monitoring_minutes);

    // Only calculate score if we have sufficient data
    if (monitoring_minutes < 1.0) return 0.0;

    // Calculate deviation from baseline
    if (hour_rate < 0.1) hour_rate = 0.1; // Minimum baseline
    double deviation = (last_min - hour_rate) / hour_rate;

    return std::min(1.0, std::max(0.0, 0.5 + deviation / 2.0));
}

void GraphNode::cleanup_time_windows() {
    auto now = std::chrono::system_clock::now();
    auto minute_ago = now - std::chrono::minutes(1);
    auto hour_ago = now - std::chrono::hours(1); {
        std::lock_guard<std::mutex> lock(temporal.window_mutex);

        // Clean minute window
        while (!temporal.minute_window.empty() &&
               temporal.minute_window.front() < minute_ago) {
            temporal.minute_window.pop();
        }

        // Clean hour window
        while (!temporal.hour_window.empty() &&
               temporal.hour_window.front() < hour_ago) {
            temporal.hour_window.pop();
        }
    }
}

int GraphNode::get_connections_last_minute() const {
    std::lock_guard<std::mutex> lock(temporal.window_mutex);
    return temporal.minute_window.size();
}

int GraphNode::get_connections_last_hour() const {
    std::lock_guard<std::mutex> lock(temporal.window_mutex);
    return temporal.hour_window.size();
}
