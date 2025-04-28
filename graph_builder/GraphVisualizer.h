//
// Created by lu on 4/28/25.
//

#ifndef GRAPH_VISUALIZER_H
#define GRAPH_VISUALIZER_H

#include "GraphBuilder.h"
#include <graphviz/gvc.h>
#include <string>
#include <memory>

#include "GraphBuilder.h"

class GraphVisualizer {
public:
    GraphVisualizer();
    ~GraphVisualizer();

    void visualize(const TrafficGraph& graph,
                  const std::string& output_file = "graph.png",
                  bool open_image = true,  bool export_cond = true);

private:
    GVC_t* gvc;

    void add_nodes(Agraph_t* graph, const TrafficGraph& traffic_graph);
    void add_edges(Agraph_t* graph, const TrafficGraph& traffic_graph);
    void apply_default_styles(Agraph_t* graph);
    std::string generate_node_id(const std::string& original_id);
    void export_to_dot(const TrafficGraph& graph, const std::string& filename);
};

#endif // GRAPH_VISUALIZER_H