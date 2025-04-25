//
// Created by lu on 4/25/25.
//

#include "TrafficAnalyzer.h"

    static std::vector<std::string> detect_suspicious_activity(const TrafficGraph& graph) {
        std::vector<std::string> anomalies;

        // Example: Detect port scanning
        auto nodes = graph.get_nodes();
        for (const auto& node : nodes) {
            if (node->type == "host") {
                // Check for many connections to different ports
                // Implement your detection logic here
            }
        }

        return anomalies;
    }

    static std::vector<std::string> find_communication_patterns(const TrafficGraph& graph) {
        std::vector<std::string> patterns;
        // Implement pattern detection
        return patterns;
    }
