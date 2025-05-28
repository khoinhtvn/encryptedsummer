//
// Created by lu on 4/25/25.
// Modified for parallel processing.
//

#include "includes/ZeekLogParser.h"
#include "includes/GraphBuilder.h" // Assuming this exists
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
    running_ = true;
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
    monitor_directory();
}

void ZeekLogParser::stop_monitoring() {
    running_ = false;
    entry_queue_.stop();
    for (auto &thread : monitor_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    for (auto &thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void ZeekLogParser::monitor_directory() {
    while (running_) {
        std::vector<std::string> current_files;
        for (const auto &entry : fs::directory_iterator(log_directory_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".log") {
                current_files.push_back(entry.path().string());
            }
        }

        {
            std::lock_guard<std::mutex> lock(tracked_files_mutex_);
            for (const auto &file_path : current_files) {
                if (tracked_files_.find(file_path) == tracked_files_.end()) {
                    tracked_files_[file_path] = FileState(file_path);
                    monitor_threads_.emplace_back(&ZeekLogParser::monitor_single_file, this, file_path);
                } else {
                    // Check if the file has been modified (size change)
                    FileState current_state(file_path);
                    if (current_state.update() && current_state.last_size != tracked_files_[file_path].last_size) {
                        monitor_threads_.emplace_back(&ZeekLogParser::monitor_single_file, this, file_path);
                        tracked_files_[file_path] = current_state;
                    }
                }
            }
            // Remove monitors for files that no longer exist (optional, for cleanup)
            std::vector<std::string> to_remove;
            for (const auto& pair : tracked_files_) {
                bool found = false;
                for (const auto& current_file : current_files) {
                    if (pair.first == current_file) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    to_remove.push_back(pair.first);
                }
            }
            for (const auto& file_path : to_remove) {
                tracked_files_.erase(file_path);
                // Need to handle stopping the monitoring thread for this file if implemented
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void ZeekLogParser::monitor_single_file(const std::string& file_path) {
    std::map<std::string, FileState> local_tracked_state;
    std::map<std::string, std::string> local_partial_lines;
    FileState current_state(file_path);
    if (current_state.update()) {
        local_tracked_state[file_path] = current_state;
        process_log_file(file_path); // Process the entire file initially
    }

    while (running_) {
        FileState new_state(file_path);
        if (!new_state.update()) {
            std::cerr << "[Monitor] Error accessing " << file_path << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        if (local_tracked_state.count(file_path)) {
            const auto& old_state = local_tracked_state[file_path];
            if (!(old_state == new_state)) {
                std::cout << "[Monitor] File changed (rotated/recreated): " << file_path << std::endl;
                process_log_file(file_path);
                local_tracked_state[file_path] = new_state;
                local_partial_lines[file_path].clear();
            } else if (new_state.last_size > old_state.last_size) {
                handle_appended_data(file_path, old_state.last_size, new_state.last_size);
                local_tracked_state[file_path] = new_state;
            }
        } else {
            std::cout << "[Monitor] New file detected by single file monitor: " << file_path << std::endl;
            process_log_file(file_path);
            local_tracked_state[file_path] = new_state;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
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
            // For multi-valued types, you might need to merge instead of overwrite
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
    // Initialize with default values (optional, but good practice)
    feature_map["timestamp"] = "0.0";
    feature_map["src_ip"] = "";
    feature_map["src_port"] = "0";
    feature_map["dst_ip"] = "";
    feature_map["dst_port"] = "0";
    feature_map["protocol"] = "tcp";
    feature_map["service"] = "UNKNOWN";
    feature_map["orig_bytes"] = "0";
    feature_map["resp_bytes"] = "0";
    feature_map["conn_state"] = "UNKNOWN";
    feature_map["local_orig"] = "false";
    feature_map["local_resp"] = "false";
    feature_map["history"] = "";
    feature_map["orig_pkts"] = "0";
    feature_map["resp_pkts"] = "0";
    feature_map["orig_ip_bytes"] = "0";
    feature_map["resp_ip_bytes"] = "0";
    feature_map["ssl_version"] = "";
    feature_map["ssl_cipher"] = "";
    feature_map["ssl_curve"] = "";
    feature_map["ssl_server_name"] = "";
    feature_map["ssl_resumed"] = "false";
    feature_map["ssl_last_alert"] = "";
    feature_map["ssl_next_protocol"] = "";
    feature_map["ssl_established"] = "false";
    feature_map["ssl_history"] = "";
    feature_map["ssl_cert_chain_fps"] = "";
    feature_map["ssl_client_cert_chain_fps"] = "";
    feature_map["ssl_sni_matches_cert"] = "false";
    feature_map["http_method"] = "UNKNOWN";
    feature_map["host"] = "";
    feature_map["uri"] = "";
    feature_map["referrer"] = "";
    feature_map["http_version"] = "";
    feature_map["http_user_agent"] = "Unknown";
    feature_map["origin"] = "";
    feature_map["http_status_code"] = "";
    feature_map["username"] = "";

    if (combined_data.count("conn")) {
        const auto& conn_data = combined_data.at("conn");
        try {
            feature_map["timestamp"] = conn_data.count("ts") ? conn_data.at("ts") : "0.0";
            feature_map["src_ip"] = conn_data.count("id.orig_h") ? conn_data.at("id.orig_h") : "";
            feature_map["src_port"] = conn_data.count("id.orig_p") ? conn_data.at("id.orig_p") : "0";
            feature_map["dst_ip"] = conn_data.count("id.resp_h") ? conn_data.at("id.resp_h") : "";
            feature_map["dst_port"] = conn_data.count("id.resp_p") ? conn_data.at("id.resp_p") : "0";
            feature_map["protocol"] = conn_data.count("proto") ? conn_data.at("proto") : "tcp";
            feature_map["service"] = conn_data.count("service") ? conn_data.at("service") : "";
            feature_map["orig_bytes"] = conn_data.count("orig_bytes") && conn_data.at("orig_bytes") != "-" ? conn_data.at("orig_bytes") : "0";
            feature_map["resp_bytes"] = conn_data.count("resp_bytes") && conn_data.at("resp_bytes") != "-" ? conn_data.at("resp_bytes") : "0";
            feature_map["conn_state"] = conn_data.count("conn_state") ? conn_data.at("conn_state") : "";
            feature_map["local_orig"] = conn_data.count("local_orig") ? (conn_data.at("local_orig") == "T" ? "true" : "false") : "false";
            feature_map["local_resp"] = conn_data.count("local_resp") ? (conn_data.at("local_resp") == "T" ? "true" : "false") : "false";
            feature_map["history"] = conn_data.count("history") ? conn_data.at("history") : "";
            feature_map["orig_pkts"] = conn_data.count("orig_pkts") && conn_data.at("orig_pkts") != "-" ? conn_data.at("orig_pkts") : "0";
            feature_map["resp_pkts"] = conn_data.count("resp_pkts") && conn_data.at("resp_pkts") != "-" ? conn_data.at("resp_pkts") : "0";
            feature_map["orig_ip_bytes"] = conn_data.count("orig_ip_bytes") && conn_data.at("orig_ip_bytes") != "-" ? conn_data.at("orig_ip_bytes") : "0";
            feature_map["resp_ip_bytes"] = conn_data.count("resp_ip_bytes") && conn_data.at("resp_ip_bytes") != "-" ? conn_data.at("resp_ip_bytes") : "0";
        } catch (const std::exception& e) {
            std::cerr << "[ZeekLogParser] Error parsing conn data for UID: " << uid << " - " << e.what() << std::endl;
            return; // Skip this entry if parsing fails
        }
    }

    if (combined_data.count("ssl")) {
        const auto& ssl_data = combined_data.at("ssl");
        feature_map["ssl_version"] = ssl_data.count("version") ? ssl_data.at("version") : "";
        feature_map["ssl_cipher"] = ssl_data.count("cipher") ? ssl_data.at("cipher") : "";
        feature_map["ssl_curve"] = ssl_data.count("curve") ? ssl_data.at("curve") : "";
        feature_map["ssl_server_name"] = ssl_data.count("server_name") ? ssl_data.at("server_name") : "";
        feature_map["ssl_resumed"] = ssl_data.count("resumed") ? (ssl_data.at("resumed") == "T" ? "true" : "false") : "false";
        feature_map["ssl_last_alert"] = ssl_data.count("last_alert") ? ssl_data.at("last_alert") : "";
        feature_map["ssl_next_protocol"] = ssl_data.count("next_protocol") ? ssl_data.at("next_protocol") : "";
        feature_map["ssl_established"] = ssl_data.count("established") ? (ssl_data.at("established") == "T" ? "true" : "false") : "false";
        feature_map["ssl_history"] = ssl_data.count("ssl_history") ? ssl_data.at("ssl_history") : "";
        feature_map["ssl_cert_chain_fps"] = ssl_data.count("cert_chain_fps") ? ssl_data.at("cert_chain_fps") : "";
        feature_map["ssl_client_cert_chain_fps"] = ssl_data.count("client_cert_chain_fps") ? ssl_data.at("client_cert_chain_fps") : "";
        feature_map["ssl_sni_matches_cert"] = ssl_data.count("sni_matches_cert") ? (ssl_data.at("sni_matches_cert") == "T" ? "true" : "false") : "false";
    }

    if (combined_data.count("http")) {
        const auto& http_data = combined_data.at("http");
        feature_map["http_method"] = http_data.count("method") ? http_data.at("method") : "";
        feature_map["host"] = http_data.count("host") ? http_data.at("host") : "";
        feature_map["uri"] = http_data.count("uri") ? http_data.at("uri") : "";
        feature_map["referrer"] = http_data.count("referrer") ? http_data.at("referrer") : "";
        feature_map["http_version"] = http_data.count("version") ? http_data.at("version") : "";
        feature_map["http_user_agent"] = http_data.count("user_agent") ? http_data.at("user_agent") : "";
        feature_map["origin"] = http_data.count("origin") ? http_data.at("origin") : "";
        feature_map["http_status_code"] = http_data.count("status_code") ? http_data.at("status_code") : "";
        feature_map["username"] = http_data.count("username") ? http_data.at("username") : "";
    }

    if (!feature_map["src_ip"].empty() && !feature_map["dst_ip"].empty()) {
        FeatureEncoder encoder;
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
    // Add parsing for other log types
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

/*
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
*/
    return data;
}
void ZeekLogParser::handle_appended_data(const std::string& file_path, off_t old_size, off_t new_size) {
    std::ifstream in(file_path);
    if (!in) {
        std::cerr << "[Parser] Error opening file for appended data: " << file_path << std::endl;
        return;
    }
    in.seekg(old_size);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    process_content(file_path, content);
}