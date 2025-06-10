import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from matplotlib.gridspec import GridSpec
def create_gat_details_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Remove axes
    for ax in [ax1, ax2]:
        ax.axis('off')

    # First GAT Layer
    ax1.set_title("GAT Layer 1 (Input → 64 dimensions)", fontsize=12, pad=20)

    # Input features
    ax1.text(0.5, 0.9, "Input Features", ha='center', va='center', fontsize=10)
    for i in range(5):
        ax1.add_patch(Rectangle((0.4 + i * 0.04, 0.8), 0.03, 0.05,
                                ec="gray", fc="lightgray"))

    # Attention mechanism
    ax1.text(0.5, 0.7, "Multi-head Attention (4 heads)", ha='center', va='center', fontsize=10)
    for i in range(4):
        ax1.add_patch(plt.Circle((0.4 + i * 0.07, 0.6), 0.025, fc="#f39c12", ec="none"))
        ax1.text(0.4 + i * 0.07, 0.6, str(i + 1), ha='center', va='center',
                 fontsize=8, color='white')

    # Processing
    ax1.text(0.5, 0.5, "Concatenation + Transformation", ha='center', va='center', fontsize=10)
    ax1.arrow(0.5, 0.45, 0, -0.05, head_width=0.03, head_length=0.02, fc='k')

    # Output features
    ax1.text(0.5, 0.35, "64-dimensional Output", ha='center', va='center', fontsize=10)
    for i in range(8):
        ax1.add_patch(Rectangle((0.3 + i * 0.05, 0.25), 0.04, 0.05,
                                ec="gray", fc="lightblue"))

    # Add regularization info
    ax1.text(0.5, 0.15, "Batch Normalization + ELU + Dropout (20%)",
             ha='center', va='center', fontsize=9, style='italic')

    # Second GAT Layer
    ax2.set_title("GAT Layer 2 (64 → 32 dimensions)", fontsize=12, pad=20)

    # Input from previous layer
    ax2.text(0.5, 0.9, "64-dimensional Input", ha='center', va='center', fontsize=10)
    for i in range(8):
        ax2.add_patch(Rectangle((0.3 + i * 0.05, 0.8), 0.04, 0.05,
                                ec="gray", fc="lightblue"))

    # Single attention head
    ax2.text(0.5, 0.7, "Single Attention Head", ha='center', va='center', fontsize=10)
    ax2.add_patch(plt.Circle((0.5, 0.6), 0.03, fc="#f39c12", ec="none"))
    ax2.text(0.5, 0.6, "1", ha='center', va='center', fontsize=8, color='white')

    # Processing
    ax2.text(0.5, 0.5, "Transformation + Residual", ha='center', va='center', fontsize=10)
    ax2.arrow(0.5, 0.45, 0, -0.05, head_width=0.03, head_length=0.02, fc='k')

    # Output features
    ax2.text(0.5, 0.35, "32-dimensional Embeddings", ha='center', va='center', fontsize=10)
    for i in range(4):
        ax2.add_patch(Rectangle((0.4 + i * 0.05, 0.25), 0.04, 0.05,
                                ec="gray", fc="lightgreen"))

    # Add features
    ax2.text(0.5, 0.15, "Edge Feature Integration + Final Embeddings",
             ha='center', va='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig('gat_layer_details.pdf', bbox_inches='tight', dpi=300)
    plt.close()


create_gat_details_figure()