cmake_minimum_required(VERSION 3.31)
project(graph_builder)

set(CMAKE_CXX_STANDARD 20)

add_executable(graph_builder main.cpp
        ZeekLogParser.cpp
        ZeekLogParser.h
        GraphBuilder.cpp
        GraphBuilder.h
        LogMonitor.cpp
        LogMonitor.h
        TrafficAnalyzer.cpp
        TrafficAnalyzer.h)
