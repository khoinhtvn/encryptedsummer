/**
 * @file ZeekLogParser.cpp
 * @brief Implementation file for the ZeekLogParser class.
 */

#include "includes/ZeekLogParser.h"
#include "includes/GraphBuilder.h"
#include "includes/EdgeFeatureEncoder.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <unordered_set>

// Define and initialize global constant strings
const std::string DEFAULT_TIMESTAMP = "0.0";
const std::string DEFAULT_EMPTY_STRING = "";
const std::string DEFAULT_PORT = "0";
const std::string DEFAULT_PROTOCOL = "tcp";
const std::string DEFAULT_SERVICE = "UNKNOWN";
const std::string DEFAULT_BYTES = "0";
const std::string DEFAULT_CONN_STATE = "UNKNOWN";
const std::string DEFAULT_FALSE = "false";
const std::string DEFAULT_PKTS = "0";
const std::string DEFAULT_USER_AGENT = "Unknown";

namespace fs = std::filesystem;

bool FileState::update() {
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) != 0) {
        return false;
    }
    inode = file_stat.st_ino;
    last_size = file_stat.st_size;
    return true;
}
bool FileState::operator==(const FileState &other){
    return inode == other.inode && last_size == other.last_size;
}

void SafeQueue::enqueue(LogEntry entry) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(entry);
    condition_.notify_one();
}

LogEntry SafeQueue::dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]{ return !queue_.empty() || !running_; });
    if (!queue_.empty()) {
        LogEntry entry = queue_.front();
        queue_.pop();
        return entry;
    }
    return {}; // Return empty LogEntry if stopping and queue is empty
}

void SafeQueue::stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    condition_.notify_all();
}

void SafeQueue::stop_waiting() {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    condition_.notify_all();
}

bool SafeQueue::is_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

bool SafeQueue::is_empty() const { // New method implementation
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}


ZeekLogParser::ZeekLogParser(const std::string &log_dir)
    : log_directory_(log_dir),
      running_(false),
      processing_thread_running_(false) {} // Initialize new atomic flag

ZeekLogParser::~ZeekLogParser() {
    stop_monitoring();
}

void ZeekLogParser::start_monitoring() {
    {
        std::lock_guard<std::mutex> lock(running_mutex_);
        running_ = true;
    }

    auto monitor_new_files = [this]() {
        while (isRunning()) {
            // Discover all interesting files in the directory
            for (const auto& entry : fs::directory_iterator(log_directory_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".log" && is_interesting_log_file(entry.path().string())) {
                    std::string file_path = entry.path().string();
                    {
                        std::lock_guard<std::mutex> lock(tracked_files_mutex_);
                        if (tracked_files_.find(file_path) == tracked_files_.end()) {
                            tracked_files_[file_path] = FileState(file_path);
                            tracked_files_[file_path].processed_size = 0;
                            {
                                std::lock_guard<std::mutex> lock(monitor_threads_mutex_);
                                monitor_threads_.emplace_back(&ZeekLogParser::monitor_file, this, file_path);
                            }
                            std::cout << "[New File Monitor] Started monitoring for new file: " << file_path << std::endl;
                        }
                    }
                }
            }
            // Wait for a certain period before checking for new files again
            std::this_thread::sleep_for(std::chrono::seconds(5)); // Adjust the interval as needed
        }
        std::cout << "[New File Monitor] Exiting." << std::endl;
    };

    // Start the initial discovery and monitoring of existing files
    for (const auto &entry : fs::directory_iterator(log_directory_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".log" && is_interesting_log_file(entry.path().string())) {
            std::string file_path = entry.path().string();
            {
                std::lock_guard<std::mutex> lock(tracked_files_mutex_);
                if (tracked_files_.find(file_path) == tracked_files_.end()) {
                    tracked_files_[file_path] = FileState(file_path);
                    tracked_files_[file_path].processed_size = 0;
                    {
                        std::lock_guard<std::mutex> lock(monitor_threads_mutex_);
                        monitor_threads_.emplace_back(&ZeekLogParser::monitor_file, this, file_path);
                    }
                    std::cout << "[Main Thread] Started monitoring for: " << file_path << std::endl;
                } else {
                    std::cout << "[Main Thread] Already monitoring: " << file_path << std::endl;
                }
            }
        }
    }

    // Start a separate thread to periodically check for new files
    new_file_monitor_thread_ = std::thread(monitor_new_files);
    std::cout << "[Main Thread] Started new file monitoring thread." << std::endl;

    // Start worker threads for processing enqueued log entries
    for (int i = 0; i < num_worker_threads_; ++i) {
        worker_threads_.emplace_back([this]() {
            while (running_ || !entry_queue_.is_empty()) { // Process while running or queue is not empty
                LogEntry entry = entry_queue_.dequeue();
                if (!entry.log_type.empty()) {
                    process_entry(entry);
                } else if (!running_ && entry_queue_.is_empty()) { // If not running and queue is truly empty, break
                    break;
                }
            }
            std::cout << "[Worker Thread] Exiting." << std::endl;
        });
    }

    // Start the new processing thread
    processing_thread_running_ = true;
    processing_thread_ = std::thread(&ZeekLogParser::processing_loop, this);
    std::cout << "[Main Thread] Started processing thread." << std::endl;
}

void ZeekLogParser::stop_monitoring() {
    {
        std::unique_lock<std::mutex> lock(running_mutex_);
        running_ = false; // Signal all threads to stop
    }
    running_cv_.notify_all(); // Notify monitoring threads waiting on condition variable
    entry_queue_.stop();       // Signal the SafeQueue to stop and unblock worker threads

    // Signal processing thread to stop and join
    {
        // Lock for processing_cv_ to ensure notification is not missed
        std::lock_guard<std::mutex> lock(uid_last_update_time_mutex_);
        processing_thread_running_ = false;
    }
    processing_cv_.notify_one(); // Notify processing thread to wake up and check its running state

    // Stop and join the new file monitoring thread
    if (new_file_monitor_thread_.joinable()) {
        new_file_monitor_thread_.join();
        std::cout << "[Main Thread] Stopped new file monitoring thread." << std::endl;
    }

    // Join all monitoring threads
    {
        std::lock_guard<std::mutex> lock(monitor_threads_mutex_);
        for (auto &thread : monitor_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        monitor_threads_.clear();
    }

    // Join all worker threads
    for (auto &thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Join the processing thread
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    std::cout << "[Main Thread] Processing thread joined." << std::endl;
}

bool ZeekLogParser::isRunning() const {
    std::lock_guard<std::mutex> lock(running_mutex_);
    return running_;
}

bool ZeekLogParser::is_interesting_log_file(const std::string& filename) const {
    std::string stem = fs::path(filename).filename().stem().string();
    return stem == "conn" || stem == "ssl" || stem == "http";
}

void ZeekLogParser::monitor_file(const std::string& file_path) {
    std::cout << "[Monitor Thread] Starting monitoring for: " << file_path << std::endl;

    while (true) {
        {
            std::unique_lock<std::mutex> lock(running_mutex_);
            if (!running_) {
                break;
            }
            running_cv_.wait_for(lock, std::chrono::seconds(1), [this]{ return !running_; });
            if (!running_) {
                break;
            }
        }

        try {
            off_t current_size = fs::file_size(file_path);
            off_t last_processed_size;

            {
                std::lock_guard<std::mutex> lock(tracked_files_mutex_);
                if (tracked_files_.find(file_path) == tracked_files_.end()) {
                    std::cerr << "[Monitor Thread] File " << file_path << " no longer tracked. Exiting thread." << std::endl;
                    break;
                }
                FileState& state = tracked_files_.at(file_path);
                last_processed_size = state.processed_size;

                if (current_size > last_processed_size) {
                    std::ifstream in(file_path);
                    if (in.is_open()) {
                        in.seekg(last_processed_size);
                        std::string new_content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                        in.close();

                        process_content(file_path, new_content);
                        state.processed_size = current_size;
                    } else {
                        std::cerr << "[Monitor Thread] Error opening file for reading appended data: " << file_path << std::endl;
                    }
                } else if (current_size < last_processed_size) {
                    std::cout << "[Monitor Thread] File " << file_path << " truncated. Reprocessing from start." << std::endl;
                    process_log_file(file_path);
                    state.processed_size = current_size;
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "[Monitor Thread] Filesystem error for " << file_path << ": " << e.what() << std::endl;
            std::lock_guard<std::mutex> lock(tracked_files_mutex_);
            tracked_files_.erase(file_path);
            break;
        } catch (const std::exception& e) {
            std::cerr << "[Monitor Thread] General error for " << file_path << ": " << e.what() << std::endl;
        }
    }
    std::cout << "[Monitor Thread] Exiting for file: " << file_path << std::endl;
}

void ZeekLogParser::process_log_file(const std::string& file_path) {
    std::ifstream in(file_path);
    if (!in) {
        std::cerr << "[Parser] Error opening file: " << file_path << std::endl;
        return;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    process_content(file_path, content);
}

void ZeekLogParser::process_content(const std::string& path, const std::string& content) {
    std::lock_guard<std::mutex> lock(partial_lines_mutex_);
    std::string& buffer = partial_lines_[path];
    buffer += content;
    size_t start = 0;
    size_t end = buffer.find('\n');
    while (end != std::string::npos) {
        std::string line = buffer.substr(start, end - start);
        if (!line.empty() && line[0] != '#') {
            std::string log_type = fs::path(path).filename().stem().string();
            entry_queue_.enqueue(parse_log_entry(log_type, line));
        }
        start = end + 1;
        end = buffer.find('\n', start);
    }
    buffer = buffer.substr(start);
}

void ZeekLogParser::process_entry(const LogEntry& entry) {
    std::string uid = entry.data.count("uid") ? entry.data.at("uid") : "";
    if (!uid.empty()) {
        {
            std::lock_guard<std::mutex> lock(uid_data_mutex_);
            if (uid_data_.find(uid) == uid_data_.end()) {
                uid_data_[uid] = {};
            }
            uid_data_[uid][entry.log_type] = entry.data;
            // TODO: check support for multi valued fields
            if (entry.log_type == "http") {
                for (const auto& [key, value_set] : entry.set_data) {
                    for (const auto& item : value_set) {
                        uid_data_[uid]["http"][key] = item; // Simple overwrite for now, adjust if needed
                    }
                }
                for (const auto& [key, value_list] : entry.list_data) {
                    // Simple overwrite, consider how to handle multiple lists for the same UID
                    if (!value_list.empty()) {
                        uid_data_[uid]["http"][key] = value_list[0]; // Taking the first element for simplicity
                    }
                }
            }
        } // uid_data_mutex_ unlocked

        // Update last update time for this UID
        {
            std::lock_guard<std::mutex> lock(uid_last_update_time_mutex_);
            uid_last_update_time_[uid] = std::chrono::steady_clock::now();
        }
        processing_cv_.notify_one(); // Signal the processing thread that there's new data
    }
}

void ZeekLogParser::attempt_build_graph_node(const std::string& uid) {
    std::map<std::string, std::map<std::string, std::string>> combined_data;
    {
        std::lock_guard<std::mutex> lock(uid_data_mutex_);
        if (uid_data_.count(uid)) {
            combined_data = uid_data_[uid];
            // Remove the data after processing
            uid_data_.erase(uid);
        } else {
            return; // UID already processed or doesn't exist
        }
    }
    build_graph_node(uid, combined_data);
}

// New processing loop for time-based aggregation
void ZeekLogParser::processing_loop() {
    std::cout << "[Processing Thread] Starting." << std::endl;
    const std::chrono::seconds PROCESSING_DELAY_SECONDS(5); // How long to wait for more data for a UID

    while (processing_thread_running_) {
        std::unique_lock<std::mutex> lock(uid_last_update_time_mutex_);
        // Wait for new data or for a timeout to check for stale UIDs
        processing_cv_.wait_for(lock, std::chrono::seconds(1), [this]{ return !processing_thread_running_; });

        if (!processing_thread_running_) {
            break; // Exit if signaled to stop
        }

        auto now = std::chrono::steady_clock::now();
        std::vector<std::string> uids_to_process;

        // Find UIDs that haven't been updated recently
        for (auto it = uid_last_update_time_.begin(); it != uid_last_update_time_.end(); ) {
            if (now - it->second > PROCESSING_DELAY_SECONDS) {
                uids_to_process.push_back(it->first);
                it = uid_last_update_time_.erase(it); // Remove from tracking
            } else {
                ++it;
            }
        }
        lock.unlock(); // Unlock before calling build_graph_node to avoid deadlocks

        for (const std::string& uid : uids_to_process) {
            attempt_build_graph_node(uid);
        }
    }
    std::cout << "[Processing Thread] Exiting." << std::endl;
}

void ZeekLogParser::build_graph_node(const std::string& uid, const std::map<std::string, std::map<std::string, std::string>>& combined_data) {
    std::unordered_map<std::string, std::string> feature_map;

    feature_map["timestamp"] = DEFAULT_TIMESTAMP;
    feature_map["src_ip"] = DEFAULT_EMPTY_STRING;
    feature_map["src_port"] = DEFAULT_PORT;
    feature_map["dst_ip"] = DEFAULT_EMPTY_STRING;
    feature_map["dst_port"] = DEFAULT_PORT;
    feature_map["protocol"] = DEFAULT_PROTOCOL;
    feature_map["service"] = DEFAULT_SERVICE;
    feature_map["orig_bytes"] = DEFAULT_BYTES;
    feature_map["resp_bytes"] = DEFAULT_BYTES;
    feature_map["conn_state"] = DEFAULT_CONN_STATE;
    feature_map["local_orig"] = DEFAULT_FALSE;
    feature_map["local_resp"] = DEFAULT_FALSE;
    feature_map["history"] = DEFAULT_EMPTY_STRING;
    feature_map["orig_pkts"] = DEFAULT_PKTS;
    feature_map["resp_pkts"] = DEFAULT_PKTS;
    feature_map["orig_ip_bytes"] = DEFAULT_PKTS;
    feature_map["resp_ip_bytes"] = DEFAULT_PKTS;
    feature_map["ssl_version"] = DEFAULT_EMPTY_STRING;
    feature_map["ssl_cipher"] = DEFAULT_EMPTY_STRING;
    feature_map["ssl_curve"] = DEFAULT_EMPTY_STRING;
    feature_map["ssl_server_name"] = DEFAULT_EMPTY_STRING;
    feature_map["ssl_resumed"] = DEFAULT_FALSE;
    feature_map["ssl_last_alert"] = DEFAULT_EMPTY_STRING;
    feature_map["ssl_next_protocol"] = DEFAULT_EMPTY_STRING;
    feature_map["ssl_established"] = DEFAULT_FALSE;
    feature_map["ssl_history"] = DEFAULT_EMPTY_STRING;
    feature_map["http_method"] = DEFAULT_SERVICE;
    feature_map["host"] = DEFAULT_EMPTY_STRING;
    feature_map["uri"] = DEFAULT_EMPTY_STRING;
    feature_map["referrer"] = DEFAULT_EMPTY_STRING;
    feature_map["http_version"] = DEFAULT_EMPTY_STRING;
    feature_map["http_user_agent"] = DEFAULT_USER_AGENT;
    feature_map["origin"] = DEFAULT_EMPTY_STRING;
    feature_map["http_status_code"] = DEFAULT_EMPTY_STRING;
    feature_map["username"] = DEFAULT_EMPTY_STRING;

    if (combined_data.count("conn")) {
        const auto& conn_data = combined_data.at("conn");
        try {
            if (conn_data.count("ts")) feature_map["timestamp"] = conn_data.at("ts");
            if (conn_data.count("id.orig_h")) feature_map["src_ip"] = conn_data.at("id.orig_h");
            if (conn_data.count("id.orig_p")) feature_map["src_port"] = conn_data.at("id.orig_p");
            if (conn_data.count("id.resp_h")) feature_map["dst_ip"] = conn_data.at("id.resp_h");
            if (conn_data.count("id.resp_p")) feature_map["dst_port"] = conn_data.at("id.resp_p");
            if (conn_data.count("proto")) feature_map["protocol"] = conn_data.at("proto");
            if (conn_data.count("service")) feature_map["service"] = conn_data.at("service");
            if (conn_data.count("orig_bytes") && conn_data.at("orig_bytes") != "-") feature_map["orig_bytes"] = conn_data.at("orig_bytes");
            if (conn_data.count("resp_bytes") && conn_data.at("resp_bytes") != "-") feature_map["resp_bytes"] = conn_data.at("resp_bytes");
            if (conn_data.count("conn_state")) feature_map["conn_state"] = conn_data.at("conn_state");
            if (conn_data.count("local_orig")) feature_map["local_orig"] = (conn_data.at("local_orig") == "T" ? "true" : "false");
            if (conn_data.count("local_resp")) feature_map["local_resp"] = (conn_data.at("local_resp") == "T" ? "true" : "false");
            if (conn_data.count("history")) feature_map["history"] = conn_data.at("history");
            if (conn_data.count("orig_pkts") && conn_data.at("orig_pkts") != "-") feature_map["orig_pkts"] = conn_data.at("orig_pkts");
            if (conn_data.count("resp_pkts") && conn_data.at("resp_pkts") != "-") feature_map["resp_pkts"] = conn_data.at("resp_pkts");
            if (conn_data.count("orig_ip_bytes") && conn_data.at("orig_ip_bytes") != "-") feature_map["orig_ip_bytes"] = conn_data.at("orig_ip_bytes");
            if (conn_data.count("resp_ip_bytes") && conn_data.at("resp_ip_bytes") != "-") feature_map["resp_ip_bytes"] = conn_data.at("resp_ip_bytes");
        } catch (const std::exception& e) {
            std::cerr << "[ZeekLogParser] Error parsing conn data for UID: " << uid << " - " << e.what() << std::endl;
            return; // Skip this entry if parsing fails
        }
    }

    if (combined_data.count("ssl")) {
        const auto& ssl_data = combined_data.at("ssl");
        if (ssl_data.count("version")) feature_map["ssl_version"] = ssl_data.at("version");
        if (ssl_data.count("cipher")) feature_map["ssl_cipher"] = ssl_data.at("cipher");
        if (ssl_data.count("curve")) feature_map["ssl_curve"] = ssl_data.at("curve");
        if (ssl_data.count("server_name")) feature_map["ssl_server_name"] = ssl_data.at("server_name");
        if (ssl_data.count("resumed")) feature_map["ssl_resumed"] = (ssl_data.at("resumed") == "T" ? "true" : "false");
        if (ssl_data.count("last_alert")) feature_map["ssl_last_alert"] = ssl_data.at("last_alert");
        if (ssl_data.count("next_protocol")) feature_map["ssl_next_protocol"] = ssl_data.at("next_protocol");
        if (ssl_data.count("established")) feature_map["ssl_established"] = (ssl_data.at("established") == "T" ? "true" : "false");
        if (ssl_data.count("ssl_history")) feature_map["ssl_history"] = ssl_data.at("ssl_history");
    }

    if (combined_data.count("http")) {
        const auto& http_data = combined_data.at("http");
        if (http_data.count("method")) feature_map["http_method"] = http_data.at("method");
        if (http_data.count("host")) feature_map["host"] = http_data.at("host");
        if (http_data.count("uri")) feature_map["uri"] = http_data.at("uri");
        if (http_data.count("referrer")) feature_map["referrer"] = http_data.at("referrer");
        if (http_data.count("version")) feature_map["http_version"] = http_data.at("version");

        // Apply categorization for http_user_agent for EDGE features
        if (http_data.count("user_agent")) {
            feature_map["http_user_agent"] = categorize_user_agent_string(http_data.at("user_agent"));
        } else {
            feature_map["http_user_agent"] = DEFAULT_USER_AGENT; // Ensure it's set to a category
        }

        if (http_data.count("origin")) feature_map["origin"] = http_data.at("origin");
        if (http_data.count("status_code")) feature_map["http_status_code"] = http_data.at("status_code");
        if (http_data.count("username")) feature_map["username"] = http_data.at("username");
    }

    // Call the GraphBuilder to add the node and edge

    if (!feature_map["src_ip"].empty() && !feature_map["dst_ip"].empty()) {
        EdgeFeatureEncoder encoder;
        std::vector<float> encoded_features = encoder.encode_features(feature_map);
        GraphBuilder::get_instance().add_connection(feature_map, encoded_features);
    }
}

LogEntry ZeekLogParser::parse_log_entry(const std::string& log_type, const std::string& entry) {
    LogEntry log_entry;
    log_entry.log_type = log_type;
    std::vector<std::string> fields;
    std::stringstream ss(entry);
    std::string field;
    while (std::getline(ss, field, '\t')) {
        fields.push_back(field);
    }

    if (log_type == "conn") {
        log_entry.data = parse_conn_entry(fields);
    } else if (log_type == "ssl") {
        log_entry.data = parse_ssl_entry(fields);
    } else if (log_type == "http") {
        log_entry.data = parse_http_entry(fields, log_entry);
    }
    return log_entry;
}

std::map<std::string, std::string> ZeekLogParser::parse_conn_entry(const std::vector<std::string>& fields) {
    std::map<std::string, std::string> data;
    if (fields.size() > 20) {
        data["ts"] = fields[0];
        data["uid"] = fields[1];
        data["id.orig_h"] = fields[2];
        data["id.orig_p"] = fields[3];
        data["id.resp_h"] = fields[4];
        data["id.resp_p"] = fields[5];
        data["proto"] = fields[6];
        data["service"] = fields[7];
        data["duration"] = fields[8];
        data["orig_bytes"] = fields[9];
        data["resp_bytes"] = fields[10];
        data["conn_state"] = fields[11];
        data["local_orig"] = fields[12];
        data["local_resp"] = fields[13];
        data["missed_bytes"] = fields[14];
        data["history"] = fields[15];
        data["orig_pkts"] = fields[16];
        data["orig_ip_bytes"] = fields[17];
        data["resp_pkts"] = fields[18];
        data["resp_ip_bytes"] = fields[19];
    }
    return data;
}

std::map<std::string, std::string> ZeekLogParser::parse_ssl_entry(const std::vector<std::string>& fields) {
    std::map<std::string, std::string> data;
    if (fields.size() > 16) {
        data["ts"] = fields[0];
        data["uid"] = fields[1];
        data["id.orig_h"] = fields[2];
        data["id.orig_p"] = fields[3];
        data["id.resp_h"] = fields[4];
        data["id.resp_p"] = fields[5];
        data["version"] = fields[6];
        data["cipher"] = fields[7];
        data["curve"] = fields[8];
        data["server_name"] = fields[9];
        data["resumed"] = fields[10];
        data["last_alert"] = fields[11];
        data["next_protocol"] = fields[12];
        data["established"] = fields[13];
        data["cert_chain_fuids"] = fields[14];
        data["subject"] = fields[15];
        data["issuer"] = fields[16];
    }
    return data;
}

std::map<std::string, std::string> ZeekLogParser::parse_http_entry(const std::vector<std::string>& fields, LogEntry& log_entry) {
    std::map<std::string, std::string> data;
    if (fields.size() > 29) {
        data["ts"] = fields[0];
        data["uid"] = fields[1];
        data["id.orig_h"] = fields[2];
        data["id.orig_p"] = fields[3];
        data["id.resp_h"] = fields[4];
        data["id.resp_p"] = fields[5];
        data["trans_depth"] = fields[6];
        data["method"] = fields[7];
        data["host"] = fields[8];
        data["uri"] = fields[9];
        data["referrer"] = fields[10];
        data["version"] = fields[11];
        data["user_agent"] = fields[12];
        data["origin"] = fields[13];
        data["request_body_len"] = fields[14];
        data["response_body_len"] = fields[15];
        data["status_code"] = fields[16];
        data["status_msg"] = fields[17];
        data["info_code"] = fields[18];
        data["info_msg"] = fields[19];
        data["tags"] = fields[20];
        data["username"] = fields[21];
        data["password"] = fields[22];
        data["proxied"] = fields[23];
        // The following fields are present in the provided format but not used in the original parse_http_entry
        // data["orig_fuids"] = fields[24];
        // data["orig_filenames"] = fields[25];
        // data["orig_mime_types"] = fields[26];
        // data["resp_fuids"] = fields[27];
        // data["resp_filenames"] = fields[28];
        // data["resp_mime_types"] = fields[29];

        // Handle set type: tags
        std::stringstream ss_tags(fields[20]);
        std::string tag;
        while (std::getline(ss_tags, tag, ',')) {
            log_entry.set_data["tags"].insert(tag);
        }
    }
    return data;
}