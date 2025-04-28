//
// Created by lu on 4/25/25.
//

#ifndef LOGMONITOR_H
#define LOGMONITOR_H
#include <string>
#include <thread>

#include "ZeekLogParser.h"

class LogMonitor {
private:
    std::atomic<bool> running{false};
    std::thread monitor_thread;
    ZeekLogParser parser;
    public:
    LogMonitor(const std::string& log_dir);

    ~LogMonitor();

    void start();
    void stop();
};


#endif //LOGMONITOR_H
