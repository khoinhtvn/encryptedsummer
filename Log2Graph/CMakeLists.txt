cmake_minimum_required(VERSION 3.28)
project(Log2Graph)

set(CMAKE_CXX_STANDARD 20)

add_executable(Log2Graph main.cpp
        ZeekLogParser.cpp
        includes/ZeekLogParser.h
        GraphBuilder.cpp
        includes/GraphBuilder.h
        LogMonitor.cpp
        includes/LogMonitor.h
        GraphExporter.cpp
        includes/GraphExporter.h
        RealTimeAnomalyDetector.cpp
        includes/RealTimeAnomalyDetector.h
        includes/GraphUpdateQueue.h
        includes/GraphNode.h
        includes/GraphEdge.h
        includes/TrafficGraph.h
        GraphNode.cpp
        TrafficGraph.cpp
        FeatureEncoder.cpp
        includes/FeatureEncoder.h)


# Manually specify the Graphviz libraries
find_library(GVC_LIB gvc)
find_library(CGRC_LIB cgraph)

if (GVC_LIB AND CGRC_LIB)
    target_link_libraries(Log2Graph ${GVC_LIB} ${CGRC_LIB})
else ()
    message(FATAL_ERROR "Graphviz libraries not found. Make sure libgraphviz-dev is installed.")
endif ()