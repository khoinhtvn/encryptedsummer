cmake_minimum_required(VERSION 3.28)
project(Log2Graph)

set(CMAKE_CXX_STANDARD 20)

# Enable AddressSanitizer (ASan) for debug builds and disable optimization for profiling
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -static-libasan")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}  -static-libasan")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}  -fsanitize=address")
endif()

add_executable(Log2Graph main.cpp
        ZeekLogParser.cpp
        includes/ZeekLogParser.h
        GraphBuilder.cpp
        includes/GraphBuilder.h
        LogMonitor.cpp
        includes/LogMonitor.h
        GraphExporter.cpp
        includes/GraphExporter.h
        includes/GraphUpdateQueue.h
        includes/GraphNode.h
        includes/GraphEdge.h
        includes/TrafficGraph.h
        GraphNode.cpp
        TrafficGraph.cpp
        FeatureEncoder.cpp
        includes/FeatureEncoder.h
        GraphEdge.cpp)

