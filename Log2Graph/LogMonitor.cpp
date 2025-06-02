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

#include "includes/ZeekLogParser.h"

LogMonitor::LogMonitor(const std::string &log_dir) : parser(log_dir) {
}

LogMonitor::~LogMonitor() {
    stop();
}

void LogMonitor::start() {
    running = true;
    monitor_thread = std::thread([this]() {
        parser.start_monitoring();
    });
}

void LogMonitor::stop() {
    running = false;
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
}


