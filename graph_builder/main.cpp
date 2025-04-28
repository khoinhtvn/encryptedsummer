
#include <iostream>
#include <thread>

#include "includes/GraphBuilder.h"
#include "includes/GraphVisualizer.h"
#include "includes/LogMonitor.h"
#include "includes/TrafficAnalyzer.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <zeek_log_directory>" << std::endl;
        return 1;
    }

    GraphVisualizer visualizer;
    // Start monitoring Zeek logs
    LogMonitor monitor(argv[1]);
    monitor.start();

    // Main loop (could be replaced with a REST API or other interface)
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));

        static int counter = 0;
        if (++counter % 3 == 0) {
            auto now = std::chrono::system_clock::now();
            auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
            auto &graph = GraphBuilder::get_instance().get_graph();
            visualizer.visualize(graph, "./zeek_graph_" + std::to_string(UTC), true, false);
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

    monitor.stop();
    return 0;
}
