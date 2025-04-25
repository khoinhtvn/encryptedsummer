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
    std::cout << "Starting directory scan..." << std::endl;
    for (const auto& entry : fs::directory_iterator(log_directory)) {
        if (entry.path().extension() == ".log") {
            std::cout << "Found log file: " << entry.path() << std::endl;
            try {
                process_log_file(entry.path());
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << entry.path() << ": " << e.what() << std::endl;
            }
        }
    }
}


    void ZeekLogParser::process_log_file(const fs::path& file_path) {
    std::ifstream log_file(file_path);
    log_file.clear();
    if (!log_file) {
        std::cerr << "Failed to open: " << file_path << std::endl;
        return;
    }

    auto filename = file_path.filename().string();
    std::streampos start_pos = 0;

    // Verifica se abbiamo una posizione valida
    if (file_positions.find(filename) != file_positions.end()) {
        start_pos = file_positions[filename];
        if (start_pos == -1) {
            std::cerr << "Invalid position for " << filename << ", resetting to 0" << std::endl;
            start_pos = 0;
        }
    }

    log_file.seekg(start_pos);

    // Verifica lo stato dello stream dopo seekg
    if (!log_file.good()) {
        std::cerr << "Failed to seek to position " << start_pos << " in " << filename << std::endl;
        log_file.clear(); // Resetta gli errori
        log_file.seekg(0); // Riparti dall'inizio
    }

    std::string line;
    bool new_data = false;
    while (std::getline(log_file, line)) {
        if (line.empty() || line[0] == '#') continue;

        process_log_entry(filename, line);
        new_data = true;
    }

    // Verifica lo stato prima di salvare la posizione
    if (log_file.eof()) {
        std::streampos new_pos = log_file.tellg();
        if (new_pos != -1) {
            file_positions[filename] = new_pos;
        } else {
            std::cerr << "Warning: tellg() returned -1 for " << filename << std::endl;
            std::cerr << " good()=" << log_file.good() << '\n';
            std::cerr << " eof()=" << log_file.eof() << '\n';
            std::cerr << " fail()=" << log_file.fail() << '\n';
            std::cerr << " bad()=" << log_file.bad() << '\n';
            // Mantieni la vecchia posizione o resetta a 0
        }
    } else if (log_file.fail()) {
        std::cerr << "Error reading file " << filename << std::endl;
        log_file.clear();
    }
    }

    void ZeekLogParser::process_log_entry(const std::string& log_type, const std::string& entry) {
        // Parse different log types
        if (log_type == "conn.log") {
            process_conn_entry(entry);
        }
        // Add other log types as needed
    }

    void ZeekLogParser::process_conn_entry(const std::string& entry) {
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
        std::string dst_ip = fields[3];
        std::string src_port = fields[4];
        std::string dst_port = fields[5];
        std::string proto = fields[6];

        // Add to graph builder
        GraphBuilder::get_instance().add_connection(
            src_ip, dst_ip, proto, timestamp, std::stoi(src_port), std::stoi(dst_port)
        );
    }




