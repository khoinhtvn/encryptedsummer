import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def create_overall_architecture_figure_revised():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define a consistent color palette
    colors = {
        "Input": "#3498db",        # Blue
        "Encoder": "#27ae60",      # Green
        "Embeddings": "#f39c12",   # Orange
        "Reconstruction": "#e91e63", # Pink
        "Direct Scoring": "#9c27b0", # Purple
        "Online Learning": "#4caf50", # Light Green
        "Anomaly Detection": "#dc3545" # Red
    }

    # Helper function for drawing elegant boxes
    def draw_component_box(center_x, center_y, width, height, label, color, text_color='white', fontsize=12, icon=None):
        rect = patches.FancyBboxPatch((center_x - width/2, center_y - height/2),
                                      width, height,
                                      boxstyle="round,pad=0.6",
                                      facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)
        if icon:
            ax.text(center_x, center_y + height/2 * 0.6, icon, ha='center', va='center', fontsize=fontsize * 1.5, color=text_color)
            ax.text(center_x, center_y - height/2 * 0.2, label, ha='center', va='center', color=text_color, fontsize=fontsize, weight='bold')
        else:
            ax.text(center_x, center_y, label, ha='center', va='center', color=text_color, fontsize=fontsize, weight='bold')

    # Helper function for drawing arrows
    def draw_arrow(start_coords, end_coords, color, label=None):
        arrow = patches.Arrow(start_coords[0], start_coords[1],
                              end_coords[0] - start_coords[0], end_coords[1] - start_coords[1],
                              width=0.15, color=color, zorder=10)
        ax.add_patch(arrow)
        if label:
            mid_x = (start_coords[0] + end_coords[0]) / 2
            mid_y = (start_coords[1] + end_coords[1]) / 2
            # Offset label slightly to the side of the arrow
            if start_coords[0] == end_coords[0]: # Vertical arrow
                ax.text(mid_x + 0.3, mid_y, label, ha='left', va='center', fontsize=9, color='dimgray')
            else: # Horizontal arrow (less likely for main flow)
                ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom', fontsize=9, color='dimgray')


    # 1. Input Graph Data
    draw_component_box(5, 9, 5, 1, "Input Graph Data", colors["Input"], icon="📊")

    # Arrow: Input -> GAT Encoder
    draw_arrow((5, 8.4), (5, 7.6), colors["Input"], "Node Features & Graph Structure")

    # 2. GAT Encoder
    draw_component_box(5, 7, 5, 1, "GAT Encoder", colors["Encoder"], icon="🧠")

    # Arrow: GAT Encoder -> Node Embeddings
    draw_arrow((5, 6.4), (5, 5.6), colors["Encoder"], "Encoded Representations")

    # 3. Node Embeddings
    draw_component_box(5, 5, 5, 1, "Node Embeddings (32-D)", colors["Embeddings"], icon="🎯")

    # Arrow split for Dual Pathways
    draw_arrow((5, 4.4), (3, 3.6), colors["Embeddings"]) # To Reconstruction
    draw_arrow((5, 4.4), (7, 3.6), colors["Embeddings"]) # To Direct Scoring
    ax.text(5, 4.6, "Dual Pathways", ha='center', va='bottom', fontsize=10, color='dimgray', weight='bold')


    # 4. Reconstruction Pathway
    draw_component_box(3, 3, 3.5, 1, "Reconstruction Pathway", colors["Reconstruction"], icon="🔄")
    ax.text(3, 2.4, "(Indirect Anomaly Detection)", ha='center', va='top', fontsize=8, color='dimgray')

    # 5. Direct Scoring Pathway
    draw_component_box(7, 3, 3.5, 1, "Direct Scoring Pathway", colors["Direct Scoring"], icon="🎯")
    ax.text(7, 2.4, "(Embedding-based Detection)", ha='center', va='top', fontsize=8, color='dimgray')


    # Arrows from pathways to combined output
    draw_arrow((3, 2.4), (5, 1.6), colors["Reconstruction"])
    draw_arrow((7, 2.4), (5, 1.6), colors["Direct Scoring"])
    ax.text(5, 2.0, "Scores & Errors", ha='center', va='bottom', fontsize=9, color='dimgray')


    # 6. Final Anomaly Decision
    draw_component_box(5, 1, 5, 1, "Final Anomaly Detection", colors["Anomaly Detection"], icon="🚨")
    ax.text(5, 0.4, "(Combined Thresholding)", ha='center', va='top', fontsize=8, color='dimgray')

    ax.set_title("Figure 1: Overall GAT-based Anomaly Detection Architecture", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('gat_architecture_overview_revised.pdf', bbox_inches='tight')
    plt.close()

create_overall_architecture_figure_revised()