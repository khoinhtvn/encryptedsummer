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
        TrafficAnalyzer.h
        GraphVisualizer.cpp
        GraphVisualizer.h)


# Manually specify the Graphviz libraries
find_library(GVC_LIB gvc)
find_library(CGRC_LIB cgraph)

if (GVC_LIB AND CGRC_LIB)
    target_link_libraries(graph_builder ${GVC_LIB} ${CGRC_LIB})
else()
    message(FATAL_ERROR "Graphviz libraries not found. Make sure libgraphviz-dev is installed.")
endif()