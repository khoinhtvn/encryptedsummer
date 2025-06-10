import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def create_dual_pathway_figure_revised():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    embedding_color = "#f39c12"
    reconstruction_color = "#e91e63"
    direct_scoring_color = "#9c27b0"
    anomaly_detection_color = "#dc3545"

    def draw_pathway_box(center_x, center_y, width, height, title, icon, color):
        rect = patches.FancyBboxPatch((center_x - width/2, center_y - height/2),
                                      width, height,
                                      boxstyle="round,pad=0.6",
                                      facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(center_x, center_y + height/2 * 0.6, icon, ha='center', va='center', fontsize=20, color='white')
        ax.text(center_x, center_y - height/2 * 0.2, title, ha='center', va='center', color='white', fontsize=12, weight='bold')

    def draw_sub_layer(center_x, center_y, width, height, label, bg_color='white'):
        rect = patches.FancyBboxPatch((center_x - width/2, center_y - height/2),
                                      width, height,
                                      boxstyle="round,pad=0.2",
                                      facecolor=bg_color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(center_x, center_y, label, ha='center', va='center', fontsize=9, color='black', wrap=True)

    # Input Node Embeddings
    draw_pathway_box(5, 7.5, 4, 0.8, "Node Embeddings (32-D)", "🎯", embedding_color)
    ax.annotate("", xy=(5, 7), xytext=(5, 6.7), arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=0.4))
    ax.text(5, 6.5, "Input to Decoders", ha='center', va='top', fontsize=9, color='dimgray')

    # Split for pathways
    ax.plot([5, 3, 3], [6.4, 6.4, 6], color='gray', linestyle=':', linewidth=1)
    ax.plot([5, 7, 7], [6.4, 6.4, 6], color='gray', linestyle=':', linewidth=1)


    # --- Reconstruction Pathway ---
    rec_center_x = 3
    rec_start_y = 5.5
    draw_pathway_box(rec_center_x, 4.5, 4.5, 4.5, "Reconstruction Pathway", "🔄", reconstruction_color)

    # Decoder Layers
    draw_sub_layer(rec_center_x, 5.8, 2.5, 0.6, "Expansion Layer (32 → 128D)\nReLU + BatchNorm")
    draw_sub_layer(rec_center_x, 5.0, 2.5, 0.6, "Intermediate Layer (128 → 64D)\nReLU + Dropout")
    draw_sub_layer(rec_center_x, 4.2, 2.5, 0.6, "Output Layer (64 → Original Features)\nLinear Reconstruction")

    ax.text(rec_center_x, 2.9,
            "**Anomaly Detection:**\n"
            "• High Reconstruction Error\n"
            "• Threshold: $\\mu + 2\\sigma$ (Adaptive)",
            ha='center', va='top', fontsize=9, color='dimgray',
            bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='red', lw=0.5, alpha=0.6))
    ax.annotate("", xy=(rec_center_x, 3.4), xytext=(rec_center_x, 3.8), arrowprops=dict(facecolor=reconstruction_color, shrink=0.05, width=0.1, headwidth=0.4))


    # --- Direct Scoring Pathway ---
    mlp_center_x = 7
    mlp_start_y = 5.5
    draw_pathway_box(mlp_center_x, 4.5, 4.5, 4.5, "Direct Scoring Pathway", "🎯", direct_scoring_color)

    # MLP Layers
    draw_sub_layer(mlp_center_x, 5.8, 2.5, 0.6, "Hidden Layer 1 (32 → 64 neurons)\nReLU + BatchNorm")
    draw_sub_layer(mlp_center_x, 5.0, 2.5, 0.6, "Hidden Layer 2 (64 → 32 neurons)\nReLU + Dropout")
    draw_sub_layer(mlp_center_x, 4.2, 2.5, 0.6, "Output Layer (32 → 1 score)\nSigmoid → Probability")

    ax.text(mlp_center_x, 2.9,
            "**Anomaly Detection:**\n"
            "• High Anomaly Score\n"
            "• Threshold: $P > 0.8$ (Configurable)",
            ha='center', va='top', fontsize=9, color='dimgray',
            bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='red', lw=0.5, alpha=0.6))
    ax.annotate("", xy=(mlp_center_x, 3.4), xytext=(mlp_center_x, 3.8), arrowprops=dict(facecolor=direct_scoring_color, shrink=0.05, width=0.1, headwidth=0.4))


    # Combined Decision Output
    ax.plot([rec_center_x, 5, mlp_center_x], [2.4, 2.4, 2.4], color='gray', linestyle=':', linewidth=1)
    ax.annotate("", xy=(5, 1.8), xytext=(5, 2.4), arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=0.4))
    draw_pathway_box(5, 1.3, 4, 0.8, "Final Anomaly Decision", "🚨", anomaly_detection_color)
    ax.text(5, 0.7, "Combined logic for robust detection", ha='center', va='top', fontsize=8, color='dimgray')


    ax.set_title("Figure 3: Dual Pathway Decoders and Anomaly Detection", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('dual_pathway_detection_revised.pdf', bbox_inches='tight')
    plt.close()

create_dual_pathway_figure_revised()