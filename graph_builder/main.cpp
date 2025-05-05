/**
 * @file main.cpp
 * @brief Main entry point for the Zeek log analysis and anomaly detection application.
 *
 * This application monitors Zeek logs from a specified directory, builds a graph
 * representation of the network traffic, visualizes the graph periodically,
 * and detects real-time anomalies.
 */
#include <iostream>
#include <thread>

#include "includes/GraphBuilder.h"
#include "includes/GraphVisualizer.h"
#include "includes/LogMonitor.h"
#include "includes/RealTimeAnomalyDetector.h"

/**
 * @brief Main function of the application.
 *
 * This function parses command-line arguments, initializes the log monitor,
 * anomaly detector, and graph visualizer, and then enters a main loop for
 * continuous monitoring and analysis.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments. The first argument should be the
 * path to the Zeek log directory.
 * @return 0 if the application runs successfully, 1 otherwise.
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <zeek_log_directory>" << std::endl;
        return 1;
    }

    /**
     * @brief Instance of the real-time anomaly detector.
     */
    RealTimeAnomalyDetector detector;
    /**
     * @brief Instance of the graph visualizer.
     */
    GraphVisualizer visualizer;
    /**
     * @brief Instance of the log monitor, responsible for reading and processing Zeek logs.
     *
     * The monitor is initialized with the directory containing the Zeek log files
     * provided as a command-line argument.
     */
    LogMonitor monitor(argv[1]);
    /**
     * @brief Starts the log monitoring process in a separate thread.
     */
    monitor.start();

    // Main loop (could be replaced with a REST API or other interface)
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));

        static int counter = 0;
        if (++counter % 3 == 0) {
            auto now = std::chrono::system_clock::now();
            auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
            /**
             * @brief Retrieves the singleton instance of the GraphBuilder and gets the current graph.
             */
            auto &graph = GraphBuilder::get_instance().get_graph();
            /**
             * @brief Visualizes the current network graph and saves it as a PNG file.
             *
             * The filename includes the current UTC timestamp. The 'false' argument
             * likely controls whether to clear the graph before visualizing (in this
             * context, it's probably not clearing), and 'true' might indicate
             * whether to include labels on the nodes.
             *
             * @param graph The graph to visualize.
             * @param "./zeek_graph_" + std::to_string(UTC) The base filename for the output PNG.
             * @param false Flag indicating whether to clear the graph before visualization.
             * @param true Flag indicating whether to include labels on the nodes.
             */
            visualizer.visualize(graph, "./zeek_graph_" + std::to_string(UTC), false, true);
            /**
             * @brief Detects anomalies in the current network graph.
             *
             * @param GraphBuilder::get_instance().get_graph() The graph to analyze for anomalies.
             * @return A map where the keys are the nodes identified as anomalous and the values
             * are their corresponding anomaly scores and contributing factors.
             */
            auto anomalies = detector.detect(GraphBuilder::get_instance().get_graph());
            /**
             * @brief Iterates through the detected anomalies and prints alerts for high-scoring ones.
             */
            for (const auto &[node, score]: anomalies) {
                if (score.score > 0.8) {
                    std::cout << "ALERT: " << node << " anomaly score " << score.score
                              << " (factors: ";
                    for (const auto &factor: score.contributing_factors) {
                        std::cout << factor << " ";
                    }
                    std::cout << ")\n";
                }
            }
        }

        // Perform periodic analysis
        /*
        auto& graph = GraphBuilder::get_instance().get_graph();
        auto anomalies = TrafficAnalyzer::detect_suspicious_activity(graph);

        if (!anomalies.empty()) {
            std::cout << "Detected anomalies:" << std::endl;
            for (const auto& anomaly : anomalies) {
                std::cout << " - " << anomaly << std::endl;
            }
        }*/
    }

    /**
     * @brief Stops the log monitoring process.
     */
    monitor.stop();
    return 0;
}