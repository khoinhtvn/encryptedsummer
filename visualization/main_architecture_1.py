import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})


def create_architecture_figure():
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(7, 1, figure=fig, height_ratios=[0.8, 0.8, 1.5, 0.8, 2, 0.8, 0.8])

    # Create axes
    ax_input = fig.add_subplot(gs[0])
    ax_encoder = fig.add_subplot(gs[1])
    ax_gat = fig.add_subplot(gs[2])
    ax_embed = fig.add_subplot(gs[3])
    ax_pathways = fig.add_subplot(gs[4])
    ax_online = fig.add_subplot(gs[5])
    ax_output = fig.add_subplot(gs[6])

    # Remove axes frames
    for ax in [ax_input, ax_encoder, ax_gat, ax_embed, ax_pathways, ax_online, ax_output]:
        ax.axis('off')

    # Input Section
    input_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                               boxstyle="round,pad=0.1",
                               ec="#3498db", fc="#e8f4f8", lw=2)
    ax_input.add_patch(input_box)
    ax_input.text(0.5, 0.7, "Input Graph Data", ha='center', va='center', fontsize=12, weight='bold')
    ax_input.text(0.5, 0.5, "Node Features: Network characteristics, system metrics", ha='center', va='center')
    ax_input.text(0.5, 0.3, "Graph Structure: Connections between nodes", ha='center', va='center')

    # Encoder Section
    encoder_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                                 boxstyle="round,pad=0.1",
                                 ec="#27ae60", fc="#f0f8e8", lw=2)
    ax_encoder.add_patch(encoder_box)
    ax_encoder.text(0.5, 0.7, "GAT Encoder", ha='center', va='center', fontsize=12, weight='bold')
    ax_encoder.text(0.5, 0.5, "Multi-layer Graph Attention Network", ha='center', va='center')

    # GAT Layers
    ax_gat.text(0.5, 0.9, "GAT Layers", ha='center', va='center', fontsize=11, weight='bold')

    # Layer 1
    layer1 = FancyBboxPatch((0.2, 0.5), 0.3, 0.35,
                            boxstyle="round,pad=0.1",
                            ec="#27ae60", fc="white", lw=1.5)
    ax_gat.add_patch(layer1)
    ax_gat.text(0.35, 0.75, "GAT Layer 1", ha='center', va='center', fontsize=10)
    ax_gat.text(0.35, 0.65, "Input → 64 dim", ha='center', va='center', fontsize=9)

    # Attention heads
    for i in range(4):
        ax_gat.add_patch(plt.Circle((0.25 + i * 0.07, 0.55), 0.015,
                                    fc="#f39c12", ec="none"))
        ax_gat.text(0.25 + i * 0.07, 0.55, str(i + 1), ha='center', va='center',
                    fontsize=6, color='white')

    # Layer 2
    layer2 = FancyBboxPatch((0.5, 0.5), 0.3, 0.35,
                            boxstyle="round,pad=0.1",
                            ec="#27ae60", fc="white", lw=1.5)
    ax_gat.add_patch(layer2)
    ax_gat.text(0.65, 0.75, "GAT Layer 2", ha='center', va='center', fontsize=10)
    ax_gat.text(0.65, 0.65, "64 → 32 dim", ha='center', va='center', fontsize=9)
    ax_gat.add_patch(plt.Circle((0.65, 0.55), 0.015, fc="#f39c12", ec="none"))
    ax_gat.text(0.65, 0.55, "1", ha='center', va='center', fontsize=6, color='white')

    # Embeddings
    embed_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                               boxstyle="round,pad=0.1",
                               ec="#f39c12", fc="#fff8e1", lw=2)
    ax_embed.add_patch(embed_box)
    ax_embed.text(0.5, 0.7, "32-Dimensional Node Embeddings", ha='center', va='center',
                  fontsize=12, weight='bold')
    ax_embed.text(0.5, 0.4, "Compressed representations capturing\nnode characteristics and relationships",
                  ha='center', va='center')

    # Pathways
    ax_pathways.text(0.25, 0.95, "Reconstruction Pathway", ha='center', va='center',
                     fontsize=11, weight='bold')
    ax_pathways.text(0.75, 0.95, "Direct Scoring Pathway", ha='center', va='center',
                     fontsize=11, weight='bold')

    # Reconstruction pathway
    recon_box = FancyBboxPatch((0.05, 0.1), 0.4, 0.8,
                               boxstyle="round,pad=0.1",
                               ec="#e91e63", fc="#fce4ec", lw=1.5)
    ax_pathways.add_patch(recon_box)

    # Layers in reconstruction
    for i, (text, y_pos) in enumerate(zip(
            ["Expansion: 32 → 128", "Intermediate: 128 → 64", "Output: 64 → Original"],
            [0.7, 0.5, 0.3]
    )):
        layer = Rectangle((0.1, y_pos - 0.08), 0.3, 0.1,
                          ec="gray", fc="white", lw=0.5)
        ax_pathways.add_patch(layer)
        ax_pathways.text(0.25, y_pos - 0.03, text, ha='center', va='center', fontsize=9)

    # MLP pathway
    mlp_box = FancyBboxPatch((0.55, 0.1), 0.4, 0.8,
                             boxstyle="round,pad=0.1",
                             ec="#9c27b0", fc="#f3e5f5", lw=1.5)
    ax_pathways.add_patch(mlp_box)

    # Layers in MLP
    for i, (text, y_pos) in enumerate(zip(
            ["Hidden: 32 → 64", "Hidden: 64 → 32", "Output: 32 → 1"],
            [0.7, 0.5, 0.3]
    )):
        layer = Rectangle((0.6, y_pos - 0.08), 0.3, 0.1,
                          ec="gray", fc="white", lw=0.5)
        ax_pathways.add_patch(layer)
        ax_pathways.text(0.75, y_pos - 0.03, text, ha='center', va='center', fontsize=9)

    # Online Learning
    online_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                                boxstyle="round,pad=0.1",
                                ec="#4caf50", fc="#e8f5e8", lw=2)
    ax_online.add_patch(online_box)
    ax_online.text(0.5, 0.7, "Online Learning & Adaptation", ha='center', va='center',
                   fontsize=12, weight='bold')
    ax_online.text(0.5, 0.4, "Replay buffer, statistical monitoring, adaptive learning",
                   ha='center', va='center')

    # Output
    output_box = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                                boxstyle="round,pad=0.1",
                                ec="#dc3545", fc="#f8d7da", lw=2)
    ax_output.add_patch(output_box)
    ax_output.text(0.5, 0.7, "Anomaly Detection Decision", ha='center', va='center',
                   fontsize=12, weight='bold')
    ax_output.text(0.5, 0.4, "Combined reconstruction error and direct scoring",
                   ha='center', va='center')

    # Add arrows between sections
    for ax1, ax2 in zip([ax_input, ax_encoder, ax_gat, ax_embed, ax_pathways, ax_online],
                        [ax_encoder, ax_gat, ax_embed, ax_pathways, ax_online, ax_output]):
        fig.add_artist(plt.Line2D([0.5, 0.5],
                                  [ax1.get_position().y0, ax2.get_position().y1],
                                  color='gray', linestyle='--', alpha=0.5))

    plt.tight_layout()
    return fig


fig = create_architecture_figure()
plt.savefig('gat_architecture_overview.pdf', bbox_inches='tight', dpi=300)
plt.close()