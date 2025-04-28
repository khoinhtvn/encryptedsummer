//
// Created by lu on 4/25/25.
//

#ifndef TRAFFICANALYZER_H
#define TRAFFICANALYZER_H
#include "GraphBuilder.h"


class TrafficAnalyzer {
public:
    static std::vector<std::string> detect_suspicious_activity(const TrafficGraph& graph) ;

    static std::vector<std::string> find_communication_patterns(const TrafficGraph& graph);
};
#endif //TRAFFICANALYZER_H
