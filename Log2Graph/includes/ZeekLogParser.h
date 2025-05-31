/**
 * @file ZeekLogParser.h
 * @brief Header file for the ZeekLogParser class, responsible for parsing Zeek log files in parallel.
 *
 * This file defines the `ZeekLogParser` class, which monitors a directory for Zeek log files,
 * and uses multiple threads to parse them in parallel. It also defines the `FileState` struct
 * to track the state of monitored log files and the `LogEntry` struct to hold parsed data.
 */

// Created by lu on 4/25/25.
// Modified for parallel processing.

#ifndef ZEEKLOGPARSER_H
#define ZEEKLOGPARSER_H

#include <string>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <chrono> // Add for std::chrono
#include <atomic> // Add for std::atomic

// Global constant strings for default feature values
extern const std::string DEFAULT_TIMESTAMP;
extern const std::string DEFAULT_EMPTY_STRING;
extern const std::string DEFAULT_PORT;
extern const std::string DEFAULT_PROTOCOL;
extern const std::string DEFAULT_SERVICE;
extern const std::string DEFAULT_BYTES;
extern const std::string DEFAULT_CONN_STATE;
extern const std::string DEFAULT_FALSE;
extern const std::string DEFAULT_PKTS;
extern const std::string DEFAULT_USER_AGENT;
/**
 * @brief Represents the state of a monitored file.
 *
 * This struct stores information about a file, including its inode, last known size,
 * and path.  It is used to track changes to log files over time.
 */
struct FileState {
    /**
     * @brief Unique identifier of the file (inode).
     */
    ino_t inode;
    /**
     * @brief Last known size of the file in bytes.
     */
    off_t last_size;
    /**
     * @brief The size of the file that has already been processed.
     */
    off_t processed_size;
    /**
     * @brief Full path to the file.
     */
    std::string path;

    /**
     * @brief Default constructor for FileState.
     */
    FileState() = default;

    /**
     * @brief Constructor for FileState.
     *
     * Initializes the FileState with the file's path and updates its inode and size.
     * The `processed_size` is initialized to the current `last_size` upon creation,
     * assuming the file is fully processed at the start.
     *
     * @param p The path to the file.
     */
    FileState(const std::string& p) : path(p), last_size(0), processed_size(0), inode(0) {
        update();
        processed_size = last_size; // Mark entire file as processed initially
    }

    /**
     * @brief Updates the file's inode and size.
     *
     * Retrieves the current inode and size of the file and updates the corresponding
     * members of the FileState object.
     *
     * @return true if the update was successful, false otherwise.
     */
    bool update();

    bool operator==(const FileState &other);

    /**
     * @brief Equality operator for FileState
     *
     * Compares two FileState objects for equality based on their members
     *
     * @param other The FileState object to compare with
     * @return true if the two objects are equal, false otherwise
     */
    bool operator==(const FileState& other) const;
};

/**
 * @brief Represents a parsed log entry.
 */
struct LogEntry {
    std::string log_type;
    std::map<std::string, std::string> data;
    std::map<std::string, std::vector<std::string>> list_data; // For vector types
    std::map<std::string, std::set<std::string>> set_data;      // For set types
};

/**
 * @brief A thread-safe queue for passing parsed log entries.
 */
class SafeQueue {
public:
    void enqueue(LogEntry entry);
    LogEntry dequeue();
    void stop();
    bool is_running() const;
    bool is_empty() const; // New method
private:
    std::queue<LogEntry> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool running_ = true;
};

/**
 * @brief Class responsible for parsing Zeek log files in parallel.
 *
 * The `ZeekLogParser` class monitors a directory for Zeek log files, and uses a pool
 * of worker threads to parse the log entries and enqueue them for further processing.
 */
class ZeekLogParser {
public:
    /**
     * @brief Constructor for ZeekLogParser.
     *
     * Initializes the parser with the directory containing the Zeek log files and starts
     * the monitoring and processing threads.
     *
     * @param log_dir The path to the directory containing the Zeek log files.
     */
    explicit ZeekLogParser(const std::string& log_dir);

    /**
     * @brief Destructor for ZeekLogParser.
     *
     * Stops all worker threads.
     */
    ~ZeekLogParser();

    /**
     * @brief Starts monitoring the log directory and processing files.
     *
     * This method discovers interesting log files at startup and launches a dedicated
     * monitoring thread for each. It also starts the worker threads for processing.
     */
    void start_monitoring();

    /**
     * @brief Stops the log monitoring and processing.
     *
     * Signals all monitoring and worker threads to stop and waits for them to join.
     */
    void stop_monitoring();

private:
    /**
     * @brief Checks if a given filename corresponds to an interesting log file.
     *
     * @param filename The full path to the file.
     * @return true if the filename stem is "conn", "ssl", or "http", false otherwise.
     */
    bool is_interesting_log_file(const std::string& filename) const;

    /**
     * @brief Map of file paths to FileState objects, used to track monitored files.
     * Protected by `tracked_files_mutex_`.
     */
    std::unordered_map<std::string, FileState> tracked_files_;
    /**
     * @brief The directory where Zeek log files are located.
     */
    std::string log_directory_;
    /**
     * @brief A thread-safe queue for passing log entries to processing threads.
     */
    SafeQueue entry_queue_;
    /**
     * @brief Vector of threads responsible for continuously monitoring individual log files.
     * Protected by `monitor_threads_mutex_`.
     */
    std::vector<std::thread> monitor_threads_;
    /**
     * @brief Mutex to protect access to `monitor_threads_`.
     */
    std::mutex monitor_threads_mutex_;
    /**
     * @brief Number of worker threads for processing log entries.
     */
    static const int num_worker_threads_ = 4;
    /**
     * @brief Vector of threads responsible for processing log entries from the queue.
     */
    std::vector<std::thread> worker_threads_;
    /**
     * @brief Mutex to protect access to `tracked_files_`.
     */
    std::mutex tracked_files_mutex_;
    /**
     * @brief Map to buffer incomplete log lines for each file.
     * Protected by `partial_lines_mutex_`.
     */
    std::unordered_map<std::string, std::string> partial_lines_;
    /**
     * @brief Mutex to protect access to `partial_lines_`.
     */
    std::mutex partial_lines_mutex_;
    /**
     * @brief Flag to indicate if monitoring is running.
     * Protected by `running_mutex_`.
     */
    bool running_ = false;
    /**
     * @brief Mutex to protect access to the `running_` flag.
     */
    std::mutex running_mutex_;
    /**
     * @brief Condition variable to notify monitoring threads about changes to `running_` flag.
     */
    std::condition_variable running_cv_;

    /**
     * @brief Stores the data from different log entries, keyed by their unique identifier (UID).
     *
     * This unordered map holds a temporary aggregation of log data for each UID encountered.
     * The outer key is the UID string. The value is another map where the key is the
     * log type (e.g., "conn", "ssl", "http"), and the value is a map of field names
     * to their corresponding string values from that log entry. This structure allows
     * the system to collect information from different log files related to the same
     * network connection before building the graph representation.
     *
     * Structure:
     * {
     * "some_unique_id": {
     * "conn": { "ts": "...", "id.orig_h": "...", ... },
     * "ssl":  { "version": "...", "cipher": "...", ... },
     * "http": { "method": "...", "host": "...", ... }
     * },
     * "another_uid": { ... }
     * ...
     * }
     */
    std::unordered_map<std::string, std::map<std::string, std::map<std::string, std::string>>> uid_data_;

    /**
     * @brief Mutex to protect access to the `uid_data_` map in a multi-threaded environment.
     *
     * Because multiple worker threads can be processing log entries concurrently and
     * potentially accessing or modifying the `uid_data_` map (when adding new log entry
     * data or when attempting to build a graph node), this mutex ensures thread-safe
     * operations on the shared `uid_data_` structure, preventing race conditions and
     * data corruption.
     */
    std::mutex uid_data_mutex_;

    // New members for time-based aggregation
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> uid_last_update_time_;
    std::mutex uid_last_update_time_mutex_;
    std::thread processing_thread_;
    std::atomic<bool> processing_thread_running_;
    std::condition_variable processing_cv_;


    /**
     * @brief Monitors a single log file continuously for appended content.
     *
     * This method runs in a dedicated thread for a specific file. It periodically checks
     * the file size and processes any new data that has been appended since the last check.
     *
     * @param file_path The path to the log file to monitor.
     */
    void monitor_file(const std::string& file_path);

    /**
     * @brief Processes a single log file by reading its entire content.
     *
     * This is typically used for initial processing of a file or if a file is truncated.
     *
     * @param file_path The path to the log file.
     */
    void process_log_file(const std::string& file_path);

    /**
     * @brief Processes the content of a log file or a portion of it.
     *
     * Parses the given content, handles partial lines, and enqueues individual log entries.
     *
     * @param path The path to the log file (used for `partial_lines_` buffer).
     * @param content The content string to process.
     */
    void process_content(const std::string& path, const std::string& content);

    /**
     * @brief Processes a single log entry from the queue.
     *
     * This method is executed by worker threads. It aggregates data for a UID and
     * attempts to build a graph node when sufficient data is available.
     *
     * @param entry The log entry to process.
     */
    void process_entry(const LogEntry& entry);

    /**
     * @brief Attempts to build a graph node for a given UID after all related log entries
     * have been processed and stored in `uid_data_`.
     *
     * This method retrieves all accumulated log data for a specific UID from `uid_data_`,
     * and if data exists, it calls `build_graph_node` to create the corresponding
     * node and edge in the graph. It also removes the processed UID's data from
     * `uid_data_` to ensure that the graph node is built only once per UID within a
     * certain processing window.
     *
     * @param uid The unique identifier of the connection for which to build the graph node.
     */
    void attempt_build_graph_node(const std::string& uid);

    /**
     * @brief Builds a graph node and its associated edge using the combined data from
     * different log files for a given UID.
     *
     * This method takes the UID and a map containing data from different log types
     * (e.g., "conn", "ssl", "http") associated with that UID. It extracts relevant
     * information from each log type's data and calls the `GraphBuilder` to add a
     * connection (node and edge) to the graph. The information extracted depends
     * on the availability of data for each log type in the `combined_data`.
     *
     * @param uid The unique identifier of the connection.
     * @param combined_data A map where the key is the log type (e.g., "conn", "ssl")
     * and the value is another map containing the fields and
     * their values from that log entry.
     */
    void build_graph_node(const std::string& uid, const std::map<std::string, std::map<std::string, std::string>>& combined_data);
    /**
     * @brief Parses a single log entry string into a `LogEntry` struct.
     *
     * @param log_type The type of the log entry (e.g., "conn", "ssl").
     * @param entry The raw log entry string.
     * @return The parsed `LogEntry` struct.
     */
    LogEntry parse_log_entry(const std::string& log_type, const std::string& entry);

    /**
     * @brief Parses a connection log entry.
     *
     * @param fields Vector of fields from the log entry.
     * @return A map containing the parsed fields.
     */
    std::map<std::string, std::string> parse_conn_entry(const std::vector<std::string>& fields);

    /**
     * @brief Parses an SSL log entry.
     *
     * @param fields Vector of fields from the log entry.
     * @return A map containing the parsed fields.
     */
    std::map<std::string, std::string> parse_ssl_entry(const std::vector<std::string>& fields);

    /**
     * @brief Parses an HTTP log entry.
     *
     * @param fields Vector of fields from the log entry.
     * @param log_entry The `LogEntry` struct to populate with set and list data.
     * @return A map containing the parsed fields.
     */
    std::map<std::string, std::string> parse_http_entry(const std::vector<std::string>& fields, LogEntry& log_entry);

    // New method for the processing thread
    void processing_loop();
};

#endif // ZEEKLOGPARSER_H