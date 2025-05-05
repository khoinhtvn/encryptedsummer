/**
 * @file RealTimeAnomalyDetector.h
 * @brief Header file for the RealTimeAnomalyDetector class, responsible for detecting anomalies in network traffic in real-time.
 *
 * This file defines the `RealTimeAnomalyDetector` class, which analyzes the network traffic graph
 * to identify potentially anomalous nodes based on their connection patterns and other features.
 */

// Created by lu on 4/28/25.
//

#ifndef REALTIMEANOMALYDETECTOR_H
#define REALTIMEANOMALYDETECTOR_H

#include <string>
#include <vector>
#include <unordered_map>

#include "GraphBuilder.h" // Include the header for GraphBuilder

/**
 * @brief Class for detecting real-time anomalies in network traffic.
 *
 * The `RealTimeAnomalyDetector` class analyzes the network traffic graph constructed by the
 * `GraphBuilder` to identify nodes that exhibit unusual behavior.  It calculates an anomaly
 * score for each node and provides information about the factors contributing to the score.
 */
class RealTimeAnomalyDetector {
public:
    /**
     * @brief Structure to store the anomaly score and contributing factors for a node.
     */
    struct AnomalyScore {
        /**
         * @brief The overall anomaly score for the node (higher values indicate higher anomaly likelihood).
         */
        double score;
        /**
         * @brief A vector of strings describing the factors that contributed to the anomaly score.
         * Examples: "high connection rate", "unusual protocol mix".
         */
        std::vector<std::string> contributing_factors;
    };

    /**
     * @brief Detects anomalies in the given network traffic graph.
     *
     * This method analyzes the nodes in the graph to identify those with anomalous behavior.
     * It calculates an anomaly score for each node based on its connection patterns,
     * protocol usage, and other relevant features.
     *
     * @param graph A const reference to the TrafficGraph object representing the network traffic.
     * The method does not modify the graph.
     * @return An unordered map where the key is the node ID (string) and the value is an
     * AnomalyScore struct containing the anomaly score and contributing factors for that node.
     */
    std::unordered_map<std::string, AnomalyScore> detect(const TrafficGraph &graph);
};

#endif // REALTIMEANOMALYDETECTOR_H
