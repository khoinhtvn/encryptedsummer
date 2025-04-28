//
// Created by lu on 4/25/25.
//

#ifndef ZEEKLOGPARSER_H
#define ZEEKLOGPARSER_H
#include <string>
#include <unordered_map>
#include<filesystem>
struct FileState {
    ino_t inode;          // Identificatore unico del file
    off_t last_size;      // Ultima dimensione conosciuta
    std::string path;     // Percorso completo del file
    FileState() = default;
    FileState(const std::string& p) : path(p), last_size(0), inode(0) {
        update();
    }

    bool update();

    bool operator==(const FileState& other) const ;
};
class ZeekLogParser {
public:
    explicit ZeekLogParser(const std::string& log_dir) : log_directory(log_dir) {}

    void monitor_logs();

    void process_new_file(const FileState &file);

    void process_appended_data(const std::string &path, off_t old_size, off_t new_size);

    void process_content(const std::string &path, const std::string &content);

private:
    std::unordered_map<std::string, FileState> tracked_files;
    std::string log_directory;

    // Mappa buffer per contenuti non completi
    std::unordered_map<std::string, std::string> partial_lines;

    void process_log_file(const std::filesystem::path& file_path) ;

    void process_log_entry(const std::string& log_type, const std::string& entry);

    void process_conn_entry(const std::string& entry) ;

    void process_http_entry(const std::string& entry) ;
};


#endif //ZEEKLOGPARSER_H
