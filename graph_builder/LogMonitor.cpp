//
// Created by lu on 4/25/25.
//

#include "LogMonitor.h"
//
// Created by lu on 4/25/25.
//



#include <thread>
#include <atomic>
#include <chrono>

#include "ZeekLogParser.h"

    LogMonitor::LogMonitor(const std::string& log_dir) : parser(log_dir) {}

    LogMonitor::~LogMonitor() {
        stop();
    }

    void LogMonitor::start() {
        running = true;
        monitor_thread = std::thread([this]() {
            while (running) {
                parser.monitor_logs();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    void LogMonitor::stop() {
        running = false;
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }


