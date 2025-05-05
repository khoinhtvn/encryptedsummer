/**
 * @file ZeekLogParser.h
 * @brief Header file for the ZeekLogParser class, responsible for parsing Zeek log files.
 *
 * This file defines the `ZeekLogParser` class, which parses Zeek log files to extract
 * network traffic information. It also defines the `FileState` struct to track the
 * state of monitored log files.
 */

// Created by lu on 4/25/25.
//

#ifndef ZEEKLOGPARSER_H
#define ZEEKLOGPARSER_H
#include <string>
#include <unordered_map>
#include <filesystem>

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
     *
     * @param p The path to the file.
     */
    FileState(const std::string& p) : path(p), last_size(0), inode(0) {
        update();
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
 * @brief Class responsible for parsing Zeek log files.
 *
 * The `ZeekLogParser` class monitors a directory for Zeek log files, parses the log
 * entries, and extracts relevant information.  It handles different Zeek log formats
 * and keeps track of file state to process updates efficiently.
 */
class ZeekLogParser {
public:
    /**
     * @brief Constructor for ZeekLogParser.
     *
     * Initializes the parser with the directory containing the Zeek log files.
     *
     * @param log_dir The path to the directory containing the Zeek log files.
     */
    explicit ZeekLogParser(const std::string& log_dir) : log_directory(log_dir) {}

    /**
     * @brief Monitors the log directory for new or updated log files.
     *
     * This method continuously checks the log directory for new files or changes
     * to existing files and processes them accordingly.
     */
    void monitor_logs();

    /**
     * @brief Processes a new log file.
     *
     * This method is called when a new log file is detected.  It parses the entire
     * file and extracts relevant information.
     *
     * @param file The FileState object representing the new log file.
     */
    void process_new_file(const FileState &file);

    /**
     * @brief Processes appended data in a log file.
     *
     * This method is called when an existing log file has been appended with new data.
     * It reads and processes only the new data in the file.
     *
     * @param path The path to the log file.
     * @param old_size The previous size of the log file.
     * @param new_size The current size of the log file.
     */
    void process_appended_data(const std::string &path, off_t old_size, off_t new_size);

    /**
     * @brief Processes the content of a log file or a portion of it.
     *
     * This method parses the given content of a log file and extracts relevant
     * information from individual log entries.
     *
     * @param path The path to the log file.
     * @param content The content to process.
     */
    void process_content(const std::string &path, const std::string &content);

private:
    /**
     * @brief Map of file paths to FileState objects, used to track monitored files.
     */
    std::unordered_map<std::string, FileState> tracked_files;
    /**
     * @brief The directory where Zeek log files are located.
     */
    std::string log_directory;

    /**
     * @brief Map to buffer incomplete log lines.
     *
     * Stores partial log lines, using the file path as the key, until a complete
     * line is received.
     */
    std::unordered_map<std::string, std::string> partial_lines;

    /**
     * @brief Processes a single log file.
     *
     * This method reads the log file and calls the appropriate processing
     * functions for each log entry.
     *
     * @param file_path The path to the log file.
     */
    void process_log_file(const std::filesystem::path& file_path);

    /**
     * @brief Processes a single log entry.
     *
     * This method determines the type of log entry and calls the corresponding
     * processing function.
     *
     * @param log_type The type of the log entry (e.g., "conn", "http").
     * @param entry The log entry string.
     */
    void process_log_entry(const std::string& log_type, const std::string& entry);

    /**
     * @brief Processes a connection log entry.
     *
     * This method extracts information from a connection log entry.
     *
     * @param entry The connection log entry string.
     */
    void process_conn_entry(const std::string& entry);

    /**
     * @brief Processes an HTTP log entry.
     *
     * This method extracts information from an HTTP log entry.
     *
     * @param entry The HTTP log entry string.
     */
    void process_http_entry(const std::string& entry);
};

#endif // ZEEKLOGPARSER_H
