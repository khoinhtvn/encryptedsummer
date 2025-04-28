//
// Created by lu on 4/28/25.
//

#include "GraphVisualizer.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <graphviz/gvc.h>

GraphVisualizer::GraphVisualizer() {
    gvc = gvContext();
}

GraphVisualizer::~GraphVisualizer() {
    gvFreeContext(gvc);
}

void GraphVisualizer::visualize(const TrafficGraph& graph,
                              const std::string& output_file,
                              bool open_image, bool export_cond) {
    // Crea un nuovo grafo
    Agraph_t* g = agopen(const_cast<char*>("ZeekTraffic"), Agdirected, nullptr);

    // Applica stili predefiniti
    apply_default_styles(g);

    // Aggiungi nodi e archi
    add_nodes(g, graph);
    add_edges(g, graph);

    if (export_cond) {
        export_to_dot(graph, output_file + ".dot");
    }

    // Layout e rendering
    if (gvLayout(gvc, g, "dot") != 0) {
        std::cerr << "Graphviz layout failed!" << std::endl;
    }
    if (gvRender(gvc, g, "png", fopen((output_file + ".png").c_str(), "wb")) != 0) {
        std::cerr << "Graphviz render failed!" << std::endl;
    }
    gvFreeLayout(gvc, g);

    // Chiudi il grafo
    agclose(g);

    // Apri l'immagine se richiesto
    if (open_image) {
        std::string command = "xdg-open " + output_file+ ".png";
        system(command.c_str());
    }
}

void GraphVisualizer::add_nodes(Agraph_t* graph, const TrafficGraph& traffic_graph) {
    auto nodes = traffic_graph.get_nodes();

    for (const auto& node : nodes) {
        std::string node_id = generate_node_id(node->id);
        Agnode_t* n = agnode(graph, const_cast<char*>(node_id.c_str()), 1);

        // Imposta attributi del nodo
        agsafeset(n, const_cast<char*>("label"), const_cast<char*>(node->id.c_str()), const_cast<char*>(""));
        agsafeset(n, const_cast<char*>("shape"), const_cast<char*>("ellipse"), const_cast<char*>(""));

        // Stile in base al tipo
        if (node->type == "host") {
            agsafeset(n, const_cast<char*>("color"), const_cast<char*>("blue"), const_cast<char*>(""));
            agsafeset(n, const_cast<char*>("style"), const_cast<char*>("filled"), const_cast<char*>(""));
            agsafeset(n, const_cast<char*>("fillcolor"), const_cast<char*>("lightblue"), const_cast<char*>(""));
        } else {
            agsafeset(n, const_cast<char*>("color"), const_cast<char*>("green"), const_cast<char*>(""));
        }
    }
}

void GraphVisualizer::add_edges(Agraph_t* graph, const TrafficGraph& traffic_graph) {
    auto edges = traffic_graph.get_edges();

    for (const auto& edge : edges) {
        std::string src_id = generate_node_id(edge->source);
        std::string dst_id = generate_node_id(edge->target);

        Agedge_t* e = agedge(graph,
                            agnode(graph, const_cast<char*>(src_id.c_str()), 0),
                            agnode(graph, const_cast<char*>(dst_id.c_str()), 0),
                            const_cast<char*>(""), 1);

        // Imposta attributi dell'arco
        agsafeset(e, const_cast<char*>("label"), const_cast<char*>(edge->relationship.c_str()), const_cast<char*>(""));

        // Stile in base al protocollo
        if (edge->relationship.find("http") != std::string::npos) {
            agsafeset(e, const_cast<char*>("color"), const_cast<char*>("red"), const_cast<char*>(""));
        } else if (edge->relationship.find("dns") != std::string::npos) {
            agsafeset(e, const_cast<char*>("color"), const_cast<char*>("green"), const_cast<char*>(""));
        } else {
            agsafeset(e, const_cast<char*>("color"), const_cast<char*>("gray"), const_cast<char*>(""));
        }

        // Aggiungi tooltip con dettagli
        std::string tooltip = "Ports: " + edge->attributes.at("src_port") + "→" + edge->attributes.at("dst_port");
        agsafeset(e, const_cast<char*>("tooltip"), const_cast<char*>(tooltip.c_str()), const_cast<char*>(""));
    }
}
void GraphVisualizer::apply_default_styles(Agraph_t* graph) {
    // Stili globali del grafo
    agsafeset(graph, const_cast<char*>("overlap"), const_cast<char*>("scale"), const_cast<char*>(""));
    agsafeset(graph, const_cast<char*>("splines"), const_cast<char*>("true"), const_cast<char*>(""));
    agsafeset(graph, const_cast<char*>("rankdir"), const_cast<char*>("LR"), const_cast<char*>(""));
    agsafeset(graph, const_cast<char*>("fontname"), const_cast<char*>("Arial"), const_cast<char*>(""));
    agsafeset(graph, const_cast<char*>("fontsize"), const_cast<char*>("10"), const_cast<char*>(""));
}

std::string GraphVisualizer::generate_node_id(const std::string& original_id) {
    // Crea un ID valido per Graphviz (senza caratteri speciali)
    std::string id = original_id;
    std::replace(id.begin(), id.end(), '.', '_');
    std::replace(id.begin(), id.end(), ':', '_');
    return "node_" + id;
}

    void GraphVisualizer::export_to_dot(const TrafficGraph& graph, const std::string& filename) {
std::ofstream dot_file(filename);
dot_file << "digraph ZeekTraffic {\n";

// Aggiungi nodi
for (const auto& node : graph.get_nodes()) {
    dot_file << "  \"" << node->id << "\" [shape=ellipse];\n";
}

// Aggiungi archi
for (const auto& edge : graph.get_edges()) {
    dot_file << "  \"" << edge->source << "\" -> \"" << edge->target
             << "\" [label=\"" << edge->relationship << "\"];\n";
}

dot_file << "}\n";
dot_file.close();
}