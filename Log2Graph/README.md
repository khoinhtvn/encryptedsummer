# Zeek Traffic Graph Analyzer

![Project Logo](https://via.placeholder.com/150x50?text=Zeek+Traffic+Graph)
*Real-time network traffic analysis using graph theory*

## Table of Contents

- [Features](#Features)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Command-Line Parameters](#Command-Line-Parameters)

## Features

- **Real-time Processing**: Continuously monitors Zeek logs
- **Graph Construction**: Builds network graphs with the following features
    - *Node level*:
        - Degree (in/out/combined)
        - Temporal features (connection in the last minute and hour)
    - *Edge level (one edge per connection)*
        - Source, target IP and port number
        - Protocol, timestamp
        - Host, Uri and HTTP method
        - Version, user agent, status code
        - Request and response body len
- **Advanced Analysis**:
    - Anomaly detection
    - Traffic pattern recognition
    - Connection profiling
- **Alert System**: Notifications for suspicious activities
- **Configurable Export Interval**: Set the frequency at which the network graph is exported.
- **Configurable Export Path**: Specify the directory for saving exported graph files.

## Requirements

```bash
# Core dependencies
- Zeek Network Security Monitor
- C++17 compatible compiler
- CMake 3.28+
```

## Installation

### Linux/macOS

```bash
git clone [https://github.com/khoinhtvn/encryptedsummer](https://github.com/khoinhtvn/encryptedsummer)
cd encryptedsummer/Log2Graph
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The command-line parameters for the Zeek Traffic Graph Analyzer are described in the Command-Line Parameters section of
the README file.

Here's that section again for your convenience:

Markdown

## Command-Line Parameters

The Zeek Traffic Graph Analyzer accepts the following command-line parameters:

| Parameter              | Description                                                                | Default Value | Example                                          |
|------------------------|----------------------------------------------------------------------------|---------------|--------------------------------------------------|
| `<zeek_log_directory>` | **Required.** The path to the directory containing the Zeek log files.     | N/A           | `./Log2Graph /var/log/zeek`                      |
| `--export-path`        | Optional. The directory where the generated graph DOT files will be saved. | `./`          | `./Log2Graph /logs --export-path /output/graphs` |
| `--export-interval`    | Optional. The interval in seconds between graph exports.                   | `60`          | `./Log2Graph /logs --export-interval 15`         |