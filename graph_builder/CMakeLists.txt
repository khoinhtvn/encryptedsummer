cmake_minimum_required(VERSION 3.28)
project(graph_builder)

set(CMAKE_CXX_STANDARD 20)

add_executable(graph_builder main.cpp
        ZeekLogParser.cpp
        includes/ZeekLogParser.h
        GraphBuilder.cpp
        includes/GraphBuilder.h
        LogMonitor.cpp
        includes/LogMonitor.h
        GraphVisualizer.cpp
        includes/GraphVisualizer.h
        RealTimeAnomalyDetector.cpp
        includes/RealTimeAnomalyDetector.h
        includes/GraphUpdateQueue.h
        includes/GraphNode.h
        includes/GraphEdge.h
        includes/TrafficGraph.h
        GraphNode.cpp
        TrafficGraph.cpp)


# Manually specify the Graphviz libraries
find_library(GVC_LIB gvc)
find_library(CGRC_LIB cgraph)

if (GVC_LIB AND CGRC_LIB)
    target_link_libraries(graph_builder ${GVC_LIB} ${CGRC_LIB})
else ()
    message(FATAL_ERROR "Graphviz libraries not found. Make sure libgraphviz-dev is installed.")
endif ()