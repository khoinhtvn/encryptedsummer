/**
 * @file LogMonitor.h
 * @brief Header file for the LogMonitor class, responsible for monitoring Zeek log files.
 *
 * This file defines the `LogMonitor` class, which continuously monitors a specified
 * directory for new Zeek log files, parses them, and processes the data.
 */

// Created by lu on 4/25/25.
//

#ifndef LOGMONITOR_H
#define LOGMONITOR_H
#include <string>
#include <thread>

#include "ZeekLogParser.h" // Assuming this is the correct header for ZeekLogParser

/**
 * @brief Class responsible for monitoring Zeek log files for new entries.
 *
 * The `LogMonitor` class runs in a separate thread, continuously scanning a directory
 * for new Zeek log files.  It uses a `ZeekLogParser` to parse the log data and
 * process it (e.g., by adding information to the network traffic graph).
 */
class LogMonitor {
private:
    /**
     * @brief Atomic boolean flag indicating whether the monitor thread is running.
     */
    std::atomic<bool> running{false};
    /**
     * @brief Thread object for the monitor thread.
     */
    std::thread monitor_thread;
    /**
     * @brief Instance of the ZeekLogParser class used to parse the log files.
     */
    ZeekLogParser parser; //  Instance, not pointer.

public:
    /**
     * @brief Constructor for the LogMonitor.
     *
     * Initializes the LogMonitor with the directory to monitor for Zeek logs.
     *
     * @param log_dir The path to the directory containing the Zeek log files.
     */
    LogMonitor(const std::string& log_dir);

    /**
     * @brief Destructor for the LogMonitor.
     *
     * Stops the monitor thread and joins it to ensure proper cleanup.
     */
    ~LogMonitor();

    /**
     * @brief Starts the log monitoring process in a separate thread.
     *
     * This method starts the monitor thread, which will continuously check for new
     * log files and process them.
     */
    void start();

    /**
     * @brief Stops the log monitoring process.
     *
     * This method sets the `running` flag to false, signaling the monitor thread to stop,
     * and then joins the thread to wait for it to finish.
     */
    void stop();
};


#endif //LOGMONITOR_H
