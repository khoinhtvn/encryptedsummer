/**
 * @file main.cpp
 * @brief Main entry point for the Zeek log analysis and anomaly detection application.
 *
 * This application monitors Zeek logs from a specified directory, builds a graph
 * representation of the network traffic and visualizes the graph periodically.
 */
#include <iostream>
#include <thread>
#include <filesystem>
#include <chrono>
#include <cstdlib> // For std::stoi

#include "includes/GraphBuilder.h"
#include "includes/GraphExporter.h"
#include "includes/LogMonitor.h"
#include "includes/RealTimeAnomalyDetector.h"

/**
 * @brief Main function of the application.
 *
 * This function parses command-line arguments, initializes the log monitor,
 * anomaly detector, and graph exporter, and then enters a main loop for
 * continuous monitoring and analysis.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments. The first argument should be the
 * path to the Zeek log directory. An optional --export-path argument specifies the export directory.
 * An optional --export-interval <seconds> argument specifies the export interval in seconds.
 * @return 0 if the application runs successfully, 1 otherwise.
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <zeek_log_directory> [--export-path <export_path>] [--export-interval <seconds>]" << std::endl;
        return 1;
    }

    std::string export_path = "./";
    int export_interval_seconds = 60; // Default export interval

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--export-path" && i + 1 < argc) {
            export_path = argv[i + 1];
            i++; // Skip the next argument as it's the value for --export-path
        } else if (std::string(argv[i]) == "--export-interval" && i + 1 < argc) {
            try {
                export_interval_seconds = std::stoi(argv[i + 1]);
                if (export_interval_seconds <= 0) {
                    std::cerr << "Error: Export interval must be a positive integer." << std::endl;
                    return 1;
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid export interval value: " << argv[i + 1] << std::endl;
                return 1;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Export interval value out of range: " << argv[i + 1] << std::endl;
                return 1;
            }
            i++; // Skip the next argument as it's the value for --export-interval
        }
    }
    // Ensure the export path exists
    if (!std::filesystem::exists(export_path)) {
        std::filesystem::create_directories(export_path);
    }
    /**
     * @brief Instance of the graph exporter.
     *
     * This object is responsible for exporting the graph data in various formats,
     * such as DOT files for visualization.
     */
    GraphExporter exporter;
    /**
     * @brief Instance of the log monitor, responsible for reading and processing Zeek logs.
     *
     * The monitor is initialized with the directory containing the Zeek log files
     * provided as a command-line argument. It continuously reads new log entries
     * and makes them available for graph building.
     */
    LogMonitor monitor(argv[1]);
    /**
     * @brief Starts the log monitoring process in a separate thread.
     *
     * This allows the application to process logs in the background without
     * blocking the main execution thread, enabling continuous monitoring.
     */
    monitor.start();


    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(export_interval_seconds));

        auto now = std::chrono::system_clock::now();
        auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

        /**
         * @brief Get a reference to the singleton instance of the GraphBuilder and its current graph.
         */
        auto &graph = GraphBuilder::get_instance().get_graph();

        /**
         * @brief Export the encoded full network graph to a DOT file.
         */
        exporter.export_full_graph_encoded_async(graph, export_path + std::filesystem::path::preferred_separator +
                                                     "nw_graph_encoded_" + std::to_string(UTC) +
                                                     ".dot");
        /*
        exporter.export_incremental_update_encoded_async(GraphBuilder::get_instance().get_last_updates(),
                                                        export_path + std::filesystem::path::preferred_separator +
                                                        "nw_graph_encoded_" + std::to_string(UTC) +
                                                        ".dot");
                                                        */

    }
    /**
     * @brief Stops the log monitoring process.
     *
     * This ensures that the log monitoring thread is properly terminated before
     * the application exits, releasing any resources it might be holding.
     */
    monitor.stop();
    return 0;
}