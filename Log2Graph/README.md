# Zeek Traffic Graph Analyzer

![Project Logo](https://via.placeholder.com/150x50?text=Zeek+Traffic+Graph)  
*Real-time network traffic analysis using graph theory*
Usage

## Table of Contents

- [Features](#Features)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Usage](#Usage)

## Features

- **Real-time Processing**: Continuously monitors Zeek logs
- **Graph Construction**: Builds interactive network graphs with the following features
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
- **Real time Visualization**:
    - Graphviz-based diagrams
- **Alert System**: Notifications for suspicious activities

## Requirements

```bash
# Core dependencies
- Zeek Network Security Monitor
- C++17 compatible compiler
- CMake 3.28+

# Visualization dependencies
- Graphviz (libgraphviz-dev)
```

## Installation

### Linux/maxOS

```bash
git clone https://github.com/khoinhtvn/encryptedsummer
cd encryptedsummer/Log2Graph
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows (Visual Studio)

```shell
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Usage

### Basic Analysis

The program checks for files in the specified path. It just processes files with known names (conn.log).

```bash
./Log2Graph /path/to/zeek/logs
```