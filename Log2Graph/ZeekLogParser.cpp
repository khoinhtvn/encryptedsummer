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
bool FileState::operator==(const FileState &other) const {
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
    return {};
}

void SafeQueue::stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    running_ = false;
    condition_.notify_all();
}

bool SafeQueue::is_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

ZeekLogParser::ZeekLogParser(const std::string &log_dir) : log_directory_(log_dir), running_(false) {}

ZeekLogParser::~ZeekLogParser() {
    stop_monitoring();
}

void ZeekLogParser::start_monitoring() {
    {
        std::lock_guard<std::mutex> lock(running_mutex_);
        running_ = true;
    }

    std::unordered_set<std::string> monitored_files; // Keep track of files already being monitored

    // Discover interesting files and launch a monitoring thread for each unique file
    for (const auto &entry : fs::directory_iterator(log_directory_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".log" && is_interesting_log_file(entry.path().string())) {
            std::string file_path = entry.path().string();
            std::lock_guard<std::mutex> lock(tracked_files_mutex_);
            if (tracked_files_.find(file_path) == tracked_files_.end()) {
                tracked_files_[file_path] = FileState(file_path);
                tracked_files_[file_path].processed_size = fs::file_size(file_path);
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

    // Start worker threads for processing enqueued log entries
    for (int i = 0; i < num_worker_threads_; ++i) {
        worker_threads_.emplace_back([this]() {
            while (running_ || !entry_queue_.is_running()) {
                LogEntry entry = entry_queue_.dequeue();
                if (!entry.log_type.empty()) {
                    process_entry(entry);
                }
            }
        });
    }
}

void ZeekLogParser::stop_monitoring() {
    {
        std::unique_lock<std::mutex> lock(running_mutex_);
        running_ = false; // Signal all threads to stop
    }
    running_cv_.notify_all(); // Notify monitoring threads waiting on condition variable
    entry_queue_.stop();      // Signal the SafeQueue to stop and unblock worker threads

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
}

bool ZeekLogParser::is_interesting_log_file(const std::string& filename) const {
    std::string stem = fs::path(filename).filename().stem().string();
    return stem == "conn" || stem == "ssl" || stem == "http";
}

void ZeekLogParser::monitor_file(const std::string& file_path) {
    std::cout << "[Monitor Thread] Starting monitoring for: " << file_path << std::endl;

    // Loop continuously as long as the parser is running
    while (true) {
        {
            std::unique_lock<std::mutex> lock(running_mutex_);
            if (!running_) {
                break; // Exit loop if parser is no longer running
            }
            // Wait for a short period or until signaled to stop
            running_cv_.wait_for(lock, std::chrono::seconds(1), [this]{ return !running_; });
            if (!running_) {
                break; // Check again after waking up from wait
            }
        }

        try {
            off_t current_size = fs::file_size(file_path);
            off_t last_processed_size;

            {
                std::lock_guard<std::mutex> lock(tracked_files_mutex_);
                // Ensure the file is still tracked (it might have been removed if deleted)
                if (tracked_files_.find(file_path) == tracked_files_.end()) {
                    std::cerr << "[Monitor Thread] File " << file_path << " no longer tracked. Exiting thread." << std::endl;
                    break;
                }
                FileState& state = tracked_files_.at(file_path);
                last_processed_size = state.processed_size;

                if (current_size > last_processed_size) {

                    std::ifstream in(file_path);
                    if (in.is_open()) {
                        in.seekg(last_processed_size); // Seek to the last processed position
                        std::string new_content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                        in.close();

                        process_content(file_path, new_content); // Process only the new content
                        state.processed_size = current_size;    // Update processed size
                    } else {
                        std::cerr << "[Monitor Thread] Error opening file for reading appended data: " << file_path << std::endl;
                    }
                } else if (current_size < last_processed_size) {
                    // File has been truncated or reset, re-process from beginning
                    std::cout << "[Monitor Thread] File " << file_path << " truncated. Reprocessing from start." << std::endl;
                    process_log_file(file_path); // Process the entire file
                    state.processed_size = current_size; // Update processed size
                }
                // If current_size == last_processed_size, no new data, do nothing.
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "[Monitor Thread] Filesystem error for " << file_path << ": " << e.what() << std::endl;
            // File might have been deleted or moved. Consider removing it from tracked_files_
            std::lock_guard<std::mutex> lock(tracked_files_mutex_);
            tracked_files_.erase(file_path);
            break; // Exit thread if file no longer accessible
        } catch (const std::exception& e) {
            std::cerr << "[Monitor Thread] General error for " << file_path << ": " << e.what() << std::endl;
        }
    }
    std::cout << "[Monitor Thread] Exiting for file: " << file_path << std::endl;
}


// The `monitor_directory` and `monitor_single_file_once` from previous versions
// are no longer needed as separate functions with the new `monitor_file` design.
// Their previous logic is now integrated into `start_monitoring` and `monitor_file`.
// Removed `handle_appended_data` as its logic is now within `monitor_file`.


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
        }
        attempt_build_graph_node(uid);
    }
}

void ZeekLogParser::attempt_build_graph_node(const std::string& uid) {
    std::map<std::string, std::map<std::string, std::string>> combined_data;
    {
        std::lock_guard<std::mutex> lock(uid_data_mutex_);
        if (uid_data_.count(uid)) {
            combined_data = uid_data_[uid];
            // Optionally remove the data after processing if you don't need it further
            uid_data_.erase(uid);
        } else {
            return;
        }
    }
    build_graph_node(uid, combined_data);
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
        if (http_data.count("user_agent")) feature_map["http_user_agent"] = http_data.at("user_agent");
        if (http_data.count("origin")) feature_map["origin"] = http_data.at("origin");
        if (http_data.count("status_code")) feature_map["http_status_code"] = http_data.at("status_code");
        if (http_data.count("username")) feature_map["username"] = http_data.at("username");
    }

    if (!feature_map["src_ip"].empty() && !feature_map["dst_ip"].empty()) {
        EdgeFeatureEncoder encoder;
        std::vector<float> encoded_features = encoder.encode_features(feature_map);
        GraphBuilder::get_instance().add_connection(feature_map, encoded_features);
    }
}

LogEntry ZeekLogParser::parse_log_entry(const std::string& log_type, const std::string& entry) {
    LogEntry log_entry;
    log_entry.log_type = log_type;
    std::istringstream iss(entry);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(iss, field, '\t')) {
        fields.push_back(field);
    }

    if (log_type == "conn" && fields.size() >= 21) {
        log_entry.data = parse_conn_entry(fields);
    } else if (log_type == "ssl" && fields.size() >= 17) {
        log_entry.data = parse_ssl_entry(fields);
    }  else if (log_type == "http" && fields.size() >= 30) {
        log_entry.data = parse_http_entry(fields, log_entry);
    }
    // Add parsing for other log types if needed
    return log_entry;
}

std::map<std::string, std::string> ZeekLogParser::parse_conn_entry(const std::vector<std::string>& fields) {
    std::map<std::string, std::string> data;
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
    data["tunnel_parents"] = fields[20];
    return data;
}

std::map<std::string, std::string> ZeekLogParser::parse_ssl_entry(const std::vector<std::string>& fields) {
    std::map<std::string, std::string> data;
    if (fields.size() > 0) data["ts"] = fields[0];
    if (fields.size() > 1) data["uid"] = fields[1];
    if (fields.size() > 2) data["id.orig_h"] = fields[2];
    if (fields.size() > 3) data["id.orig_p"] = fields[3];
    if (fields.size() > 4) data["id.resp_h"] = fields[4];
    if (fields.size() > 5) data["id.resp_p"] = fields[5];
    if (fields.size() > 6) data["version"] = fields[6];
    if (fields.size() > 7) data["cipher"] = fields[7];
    if (fields.size() > 8) data["curve"] = fields[8];
    if (fields.size() > 9) data["server_name"] = fields[9];
    if (fields.size() > 10) data["resumed"] = fields[10];
  if (fields.size() > 11) data["last_alert"] = fields[11];
    if (fields.size() > 12) data["next_protocol"] = fields[12];
    if (fields.size() > 13) data["established"] = fields[13];
    if (fields.size() > 14) data["ssl_history"] = fields[14];
    if (fields.size() > 15) data["cert_chain_fps"] = fields[15];
    if (fields.size() > 16) data["client_cert_chain_fps"] = fields[16];
    if (fields.size() > 17) {
        data["sni_matches_cert"] = fields[17];
    }
    return data;
}

std::map<std::string, std::string> ZeekLogParser::parse_http_entry(const std::vector<std::string>& fields, LogEntry& log_entry) {
    std::map<std::string, std::string> data;
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
    data["resp_mime_types"] = fields[19];

    // Handle set type: tags, proxied
    if (fields.size() > 20) {
        std::istringstream tags_ss(fields[20]);
        std::string tag;
        while (std::getline(tags_ss, tag, ',')) {
            if (!tag.empty()) {
                log_entry.set_data["tags"].insert(tag);
            }
        }
    }

    if (fields.size() > 23) {
        std::istringstream proxied_ss(fields[23]);
        std::string proxied_item;
        while (std::getline(proxied_ss, proxied_item, ',')) {
            if (!proxied_item.empty()) {
                log_entry.set_data["proxied"].insert(proxied_item);
            }
        }
    }

    // Handle vector types: orig_fuids, orig_filenames, orig_mime_types, resp_fuids, resp_filenames, resp_mime_types
    auto parse_vector_field = [&](const std::string& raw_string) {
        std::vector<std::string> result;
        std::istringstream ss(raw_string);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                result.push_back(item);
            }
        }
        return result;
    };

    if (fields.size() > 24) log_entry.list_data["orig_fuids"] = parse_vector_field(fields[24]);
    if (fields.size() > 25) log_entry.list_data["orig_filenames"] = parse_vector_field(fields[25]);
    if (fields.size() > 26) log_entry.list_data["orig_mime_types"] = parse_vector_field(fields[26]);
    if (fields.size() > 27) log_entry.list_data["resp_fuids"] = parse_vector_field(fields[27]);
    if (fields.size() > 28) log_entry.list_data["resp_filenames"] = parse_vector_field(fields[28]);
    if (fields.size() > 29) log_entry.list_data["resp_mime_types"] = parse_vector_field(fields[29]);

    return data;
}
/*
// This method is no longer used with the new monitor_file logic,
// as its functionality is integrated directly into monitor_file.
// It's kept here for completeness but can be removed if not needed elsewhere.
void ZeekLogParser::handle_appended_data(const std::string& file_path, off_t old_size, off_t new_size) {
    std::ifstream in(file_path);
    if (!in) {
        std::cerr << "[Parser] Error opening file for appended data: " << file_path << std::endl;
        return;
    }
    in.seekg(old_size);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    process_content(file_path, content);
}*/