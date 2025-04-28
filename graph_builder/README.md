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
- **Graph Construction**: Builds interactive network graphs
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
- CMake 3.21+

# Visualization dependencies
- Graphviz (libgraphviz-dev)
```
## Installation
### Linux/maxOS
```bash
git clone https://github.com/khoinhtvn/encryptedsummer
cd encryptedsummer/graph_builder
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
```bash
./main /path/to/zeek/logs
```