//
// Created by lu on 4/25/25.
//

#include "includes/LogMonitor.h"
//
// Created by lu on 4/25/25.
//

#include <thread>
#include <atomic>
#include <chrono>
#include <filesystem> // Include for filesystem operations

#include "includes/ZeekLogParser.h"

namespace fs = std::filesystem;

LogMonitor::LogMonitor(const std::string& log_dir) : parser(log_dir) {}

LogMonitor::~LogMonitor() {
    stop();
}

void LogMonitor::start() {
    running = true;
    monitor_thread = std::thread([this]() {
        while (running) {
            parser.monitor_new_files(); // Check for and start monitoring new files
            parser.continue_monitoring_existing(); // Continue monitoring existing files for updates
            std::this_thread::sleep_for(std::chrono::seconds(5)); // Check for new files every 5 seconds (adjust as needed)
        }
    });
}

void LogMonitor::stop() {
    running = false;
    parser.stop_monitoring(); // Stop the parser's monitoring threads
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
}