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
 * @return 0 if the application runs successfully, 1 otherwise.
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <zeek_log_directory> [--export-path <export_path>]" << std::endl;
        return 1;
    }

    std::string export_path = "./";
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--export-path" && i + 1 < argc) {
            export_path = argv[i + 1];
            break;
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
        std::this_thread::sleep_for(std::chrono::seconds(10));

        static int counter = 0;
        if (++counter % 3 == 0) {
            /**
             * @brief Get the current UTC timestamp for filename generation.
             */
            auto now = std::chrono::system_clock::now();
            auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

            /**
             * @brief Get a reference to the singleton instance of the GraphBuilder and its current graph.
             */
            auto &graph = GraphBuilder::get_instance().get_graph();

            /**
             * @brief Export the encoded incremental updates of the network graph to a DOT file.
             *
             * This call retrieves the latest changes to the graph since the last export
             * and saves them in an encoded format to a DOT file. The filename includes
             * the UTC timestamp to differentiate between exports. The output file is
             * created in the specified `export_path`.
             */
            exporter.export_incremental_update_encoded(GraphBuilder::get_instance().get_last_updates(),
                                                        export_path + std::filesystem::path::preferred_separator +
                                                        "nw_graph_encoded_" + std::to_string(UTC) +
                                                        ".dot");
        }
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