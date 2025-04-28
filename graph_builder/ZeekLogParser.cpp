//
// Created by lu on 4/25/25.
//

#include "ZeekLogParser.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include "GraphBuilder.h"

#include <sys/stat.h>
#include <unistd.h>


    bool FileState::update() {
        struct stat file_stat;
        if (stat(path.c_str(), &file_stat) != 0) {
            return false;
        }

        inode = file_stat.st_ino;
        last_size = file_stat.st_size;
        return true;
    }

    bool FileState::operator==(const FileState& other) const {
        return inode == other.inode;
    }
namespace fs = std::filesystem;

void ZeekLogParser::monitor_logs() {
    for (const auto& entry : fs::directory_iterator(log_directory)) {
        if (entry.path().extension() != ".log") continue;

        std::string path = entry.path().string();
        auto it = tracked_files.find(path);

        if (it == tracked_files.end()) {
            // Nuovo file rilevato
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

void ZeekLogParser::process_new_file(const FileState& file) {
    std::ifstream in(file.path, std::ios::binary);
    if (!in) return;

    std::string content((std::istreambuf_iterator<char>(in)),
                 std::istreambuf_iterator<char>());

    // Processa tutto il contenuto
    process_content(file.path, content);
    partial_lines[file.path].clear();
}

void ZeekLogParser::process_appended_data(const std::string& path, off_t old_size, off_t new_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return;

    in.seekg(old_size);  // Posiziona alla fine del contenuto precedente
    std::string new_content;
    new_content.resize(new_size - old_size);
    in.read(&new_content[0], new_size - old_size);

    process_content(path, new_content);
}

void ZeekLogParser::process_content(const std::string& path, const std::string& content) {
    std::string& buffer = partial_lines[path];
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
void  ZeekLogParser::process_log_entry(const std::string& log_type, const std::string& entry) {
    // Parse different log types
    if (log_type == "conn.log") {
        process_conn_entry(entry);
    }
    // Add other log types as needed
}

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


