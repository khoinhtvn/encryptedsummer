//
// Created by lu on 4/25/25.
//

#include "includes/ZeekLogParser.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include "includes/GraphBuilder.h"

#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>

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
    return inode == other.inode;
}

namespace fs = std::filesystem;

void ZeekLogParser::monitor_logs() {
    for (const auto &entry: fs::directory_iterator(log_directory)) {
        if (entry.path().extension() != ".log") continue;

        std::string path = entry.path().string();
        auto it = tracked_files.find(path);

        if (it == tracked_files.end()) {
            // Nuovo file rilevato
            std::cout << "New File found! " << path << std::endl;
            FileState new_file(path);
            if (new_file.update()) {
                process_new_file(new_file);
                tracked_files[path] = new_file;
            }
        } else {
            // File esistente - verifica modifiche
            FileState old_state = it->second;
            FileState new_state(path);

            if (!new_state.update()) {
                std::cerr << "Error accessing " << path << std::endl;
                continue;
            }

            if (!(old_state == new_state)) {
                // File è stato ruotato/ricreato
                process_new_file(new_state);
                tracked_files[path] = new_state;
            } else if (new_state.last_size > old_state.last_size) {
                // File è cresciuto - processa nuovi dati
                process_appended_data(path, old_state.last_size, new_state.last_size);
                tracked_files[path] = new_state;
            }
        }
    }
}

void ZeekLogParser::process_new_file(const FileState &file) {
    std::ifstream in(file.path, std::ios::binary);
    if (!in) return;

    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());

    // Processa tutto il contenuto
    process_content(file.path, content);
    partial_lines[file.path].clear();
}

void ZeekLogParser::process_appended_data(const std::string &path, off_t old_size, off_t new_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;

    in.seekg(old_size); // Posiziona alla fine del contenuto precedente
    std::string new_content;
    new_content.resize(new_size - old_size);
    in.read(&new_content[0], new_size - old_size);

    process_content(path, new_content);
}

void ZeekLogParser::process_content(const std::string &path, const std::string &content) {
    std::string &buffer = partial_lines[path];
    buffer += content;

    size_t start = 0;
    size_t end = buffer.find('\n');

    while (end != std::string::npos) {
        std::string line = buffer.substr(start, end - start);
        if (!line.empty() && line[0] != '#') {
            process_log_entry(fs::path(path).filename().string(), line);
        }
        start = end + 1;
        end = buffer.find('\n', start);
    }

    // Conserva dati non completi per il prossimo ciclo
    if (start < buffer.size()) {
        buffer = buffer.substr(start);
    } else {
        buffer.clear();
    }
}

void ZeekLogParser::process_log_entry(const std::string &log_type, const std::string &entry) {
    // Parse different log types
    if (log_type == "conn.log") {
        process_conn_entry(entry);
    }
    // Add other log types as needed
}

void ZeekLogParser::process_conn_entry(const std::string &entry) {
    try {
        // Split the tab-separated line
        std::vector<std::string> fields;
        std::string field;
        std::istringstream stream(entry);

        while (std::getline(stream, field, '\t')) {
            fields.push_back(field);
        }

        // Ensure there are enough fields (21 fields expected)
        if (fields.size() < 21) {
            std::cerr << "Error: Incomplete log entry: " << entry << std::endl;
            return;
        }

        // Extract fields safely
        double ts = std::stod(fields[0]);
        std::string uid = fields[1];
        std::string orig_h = fields[2];
        int orig_p = std::stoi(fields[3]);
        std::string resp_h = fields[4];
        int resp_p = std::stoi(fields[5]);
        std::string protocol = fields[6];
        std::string service = fields[7];
        double duration = fields[8] == "-" ? 0.0 : std::stod(fields[8]);
        int orig_bytes = fields[9] == "-" ? 0.0 : std::stoi(fields[9]);
        int resp_bytes = fields[10] == "-" ? 0.0 : std::stoi(fields[10]);
        std::string conn_state = fields[11];
        bool local_orig = fields[12] == "T";
        bool local_resp = fields[13] == "T";
        int missed_bytes = fields[14] == "-" ? 0.0 : std::stoi(fields[14]);
        std::string history = fields[15];
        int orig_pkts = fields[16] == "-" ? 0.0 : std::stoi(fields[16]);
        int orig_ip_bytes = fields[17] == "-" ? 0.0 : std::stoi(fields[17]);
        int resp_pkts = fields[18] == "-" ? 0.0 : std::stoi(fields[18]);
        int resp_ip_bytes = fields[19] == "-" ? 0.0 : std::stoi(fields[19]);

        // Call GraphBuilder
        GraphBuilder::get_instance().add_connection(
            orig_h, // src_ip
            resp_h, // dst_ip
            protocol, // proto
            service, // service
            std::to_string(ts), // timestamp as string
            orig_p, // src_port
            resp_p, // dst_port
            orig_bytes, // orig_bytes
            resp_bytes, // resp_bytes
            conn_state, // conn_state
            local_orig, // local_orig
            local_resp, // local_resp
            history, // history
            orig_pkts, // orig_pkts
            resp_pkts, // resp_pkts
            orig_ip_bytes, // orig_ip_bytes
            resp_ip_bytes // resp_ip_bytes
        );
    } catch (const std::exception &e) {
        std::cerr << "Error processing log entry: " << e.what() << " - Entry: " << entry << std::endl;
    }
}

/*
void  ZeekLogParser::process_conn_entry(const std::string& entry) {
    // Parse conn.log entry (TSV format)
    std::vector<std::string> fields;
    size_t start = 0;
    size_t end = entry.find('\t');

    while (end != std::string::npos) {
        fields.push_back(entry.substr(start, end - start));
        start = end + 1;
        end = entry.find('\t', start);
    }
    fields.push_back(entry.substr(start));

    if (fields.size() < 10) return; // Basic validation

    // Extract relevant fields
    std::string timestamp = fields[0];
    std::string uid = fields[1];
    std::string src_ip = fields[2];
    std::string src_port = fields[3];
    std::string dst_ip = fields[4];
    std::string dst_port = fields[5];
    std::string proto = fields[6];
    std::string duration = fields[7];
    std::string orig_bytes = fields[8];
    std::string resp_bytes = fields[9];
    std::string conn_state = fields[10];


    // Add to graph builder
    GraphBuilder::get_instance().add_connection(
        src_ip, dst_ip, proto, timestamp, std::stoi(src_port), std::stoi(dst_port)
    );
}

*/
/*
using json = nlohmann::json;

void ZeekLogParser::process_conn_entry(const std::string& entry) {
    try {
        json j = json::parse(entry);

        // Extract fields - Safely access and handle potential missing fields
        double ts = j.contains("ts") ? j["ts"].get<double>() : 0.0; // Or a default value
        std::string uid = j.contains("uid") ? j["uid"].get<std::string>() : "";
        std::string orig_h = j.contains("id.orig_h") ? j["id.orig_h"].get<std::string>() : "";
        int orig_p = j.contains("id.orig_p") ? j["id.orig_p"].get<int>() : 0;
        std::string resp_h = j.contains("id.resp_h") ? j["id.resp_h"].get<std::string>() : "";
        int resp_p = j.contains("id.resp_p") ? j["id.resp_p"].get<int>() : 0;
        int trans_depth = j.contains("trans_depth") ? j["trans_depth"].get<int>() : 0;
        std::string method = j.contains("method") ? j["method"].get<std::string>() : "";
        std::string host = j.contains("host") ? j["host"].get<std::string>() : "";
        std::string uri = j.contains("uri") ? j["uri"].get<std::string>() : "";
        std::string version = j.contains("version") ? j["version"].get<std::string>() : "";
        std::string user_agent = j.contains("user_agent") ? j["user_agent"].get<std::string>() : "";
        int request_body_len = j.contains("request_body_len") ? j["request_body_len"].get<int>() : 0;
        int response_body_len = j.contains("response_body_len") ? j["response_body_len"].get<int>() : 0;
        int status_code = j.contains("status_code") ? j["status_code"].get<int>() : 0;
        std::string status_msg = j.contains("status_msg") ? j["status_msg"].get<std::string>() : "";
        std::vector<std::string> tags = j.contains("tags") ? j["tags"].get<std::vector<std::string>>() : std::vector<std::string>();
        std::vector<std::string> resp_fuids = j.contains("resp_fuids") ? j["resp_fuids"].get<std::vector<std::string>>() : std::vector<std::string>();
        std::vector<std::string> resp_mime_types = j.contains("resp_mime_types") ? j["resp_mime_types"].get<std::vector<std::string>>() : std::vector<std::string>();

        std::string protocol = "tcp"; // Default
        if (j.contains("proto")) {
            protocol = j["proto"].get<std::string>();
        } else if (!method.empty()) {
            protocol = "http"; // If there's an HTTP method, assume HTTP
        }

        GraphBuilder::get_instance().add_connection(
            orig_h,
            resp_h,
            protocol, // TODO: Use determined protocol instead of hardcoded "tcp" or "http"
            std::to_string(ts),
            orig_p,
            resp_p,
            method,
            host,
            uri,
            version,
            user_agent,
            request_body_len,
            response_body_len,
            status_code,
            status_msg,
            tags,
            resp_fuids,
            resp_mime_types
        );


    } catch (const json::parse_error& e) {
        std::cerr << "Error parsing log entry: " << e.what() << " - Entry: " << entry << std::endl;
        // Handle the error appropriately (e.g., log it, skip the entry, etc.)
    }  catch (const json::type_error& e) {
        std::cerr << "Error extracting data: " << e.what() << " - Entry: " << entry << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General error processing log entry: " << e.what() << " - Entry: " << entry << std::endl;
    }
}

*/
