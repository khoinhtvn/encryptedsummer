//
// Created by lu on 4/28/25.
//

#ifndef REALTIMEANOMALYDETECTOR_H
#define REALTIMEANOMALYDETECTOR_H


#include <string>
#include <vector>
#include <unordered_map>

#include "GraphBuilder.h"

class RealTimeAnomalyDetector {
public:
    struct AnomalyScore {
        double score;
        std::vector<std::string> contributing_factors;
    };

    std::unordered_map<std::string, AnomalyScore> detect(const TrafficGraph &graph) {
        std::unordered_map<std::string, AnomalyScore> results;
        auto now = std::chrono::system_clock::now();

        for (const auto &[id, node]: graph.nodes) {
            AnomalyScore score;
            auto monitoring_duration = now - node->temporal.monitoring_start;
            double monitoring_minutes = std::chrono::duration<double>(monitoring_duration).count() / 60.0;

            // Only calculate score after sufficient monitoring time
            if (monitoring_minutes >= 1.0) {
                score.score = node->calculate_anomaly_score();

                // Dynamic threshold based on monitoring duration
                double threshold = 0.7 + (0.2 / (1 + monitoring_minutes / 60));

                if (score.score > threshold) {
                    // Identify contributing factors
                    double total_rate = node->temporal.total_connections / monitoring_minutes;
                    double recent_rate = node->temporal.connections_last_minute;

                    if (recent_rate > 2 * total_rate) {
                        score.contributing_factors.push_back(
                            "connection_rate(" + std::to_string(recent_rate) + " vs avg " +
                            std::to_string(total_rate) + ")");
                    }

                    if (node->features.protocol_counts.size() > 3) {
                        score.contributing_factors.push_back(
                            "protocols(" + std::to_string(node->features.protocol_counts.size()) + ")");
                    }
                }
            } else {
                score.score = 0.0;
                score.contributing_factors.push_back("learning (only " +
                                                     std::to_string((int) monitoring_minutes) + " mins data)");
            }

            results[id] = score;
        }

        return results;
    }
};
#endif
