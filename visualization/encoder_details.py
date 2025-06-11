import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def create_gat_encoder_detail_figure_revised():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    encoder_color = "#27ae60"
    attention_head_color = "#f39c12"

    def draw_layer_box(center_x, center_y, width, height, title, details, color='white', text_color='black'):
        rect = patches.FancyBboxPatch((center_x - width/2, center_y - height/2),
                                      width, height,
                                      boxstyle="round,pad=0.3",
                                      facecolor=color, edgecolor=encoder_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(center_x, center_y + height/2 - 0.2, title, ha='center', va='top', fontsize=11, weight='bold', color=text_color)
        ax.text(center_x, center_y - height/2 + 0.2, details, ha='center', va='bottom', fontsize=8, color='dimgray', linespacing=1.2)

    def draw_attention_heads(center_x, center_y, num_heads):
        head_radius = 0.15
        spacing = 0.4
        start_x = center_x - (num_heads - 1) * spacing / 2
        for i in range(num_heads):
            circle = patches.Circle((start_x + i * spacing, center_y), head_radius, color=attention_head_color, zorder=5)
            ax.add_patch(circle)
            ax.text(start_x + i * spacing, center_y, str(i+1), ha='center', va='center', color='white', fontsize=7, weight='bold')
        ax.text(center_x, center_y - 0.35, f"{num_heads} Attention Head{'s' if num_heads > 1 else ''}", ha='center', va='top', fontsize=8, color='dimgray')

    # Main "GAT Encoder" Label
    ax.text(5, 6.5, "GAT Encoder Architecture", ha='center', va='center', fontsize=14, weight='bold', color=encoder_color)
    ax.text(5, 6.1, "Multi-layer Graph Attention Network", ha='center', va='center', fontsize=10, color='dimgray')


    # GAT Layer 1
    draw_layer_box(2.5, 3.5, 4, 3, "GAT Layer 1",
                   "Input Features → 64 Dimensions\n"
                   "✓ Batch Normalization\n"
                   "✓ ELU Activation\n"
                   "✓ Dropout (20%)")
    draw_attention_heads(2.5, 4.2, 4)
    ax.text(2.5, 5.2, "Input", ha='center', va='bottom', fontsize=10, color='black') # Input arrow label
    ax.annotate("", xy=(2.5, 4.8), xytext=(2.5, 5.0), arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=0.4))


    # GAT Layer 2
    draw_layer_box(7.5, 3.5, 4, 3, "GAT Layer 2",
                   "64 Dimensions → 32 Dimensions\n"
                   "✓ Residual Connections\n"
                   "✓ Edge Feature Integration\n"
                   "✓ Final Embeddings")
    draw_attention_heads(7.5, 4.2, 1) # Single head for final layer typically
    ax.text(7.5, 5.2, "Aggregated Output", ha='center', va='bottom', fontsize=10, color='black') # Input arrow label
    ax.annotate("", xy=(7.5, 4.8), xytext=(7.5, 5.0), arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=0.4))


    # Arrow between layers
    ax.annotate("", xy=(5, 3.5), xytext=(4.5, 3.5), arrowprops=dict(facecolor=encoder_color, shrink=0.05, width=0.1, headwidth=0.4))
    ax.annotate("", xy=(5.5, 3.5), xytext=(5, 3.5), arrowprops=dict(facecolor=encoder_color, shrink=0.05, width=0.1, headwidth=0.4))
    ax.text(5, 3.7, "Concatenation/Aggregation", ha='center', va='bottom', fontsize=8, color='dimgray')

    # Output Arrow
    ax.annotate("", xy=(7.5, 1.8), xytext=(7.5, 1.4), arrowprops=dict(facecolor='black', shrink=0.05, width=0.1, headwidth=0.4))
    ax.text(7.5, 1.2, "32D Node Embeddings", ha='center', va='top', fontsize=10, color='black')


    ax.set_title("Figure 2: Detailed GAT Encoder Architecture", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('gat_encoder_detail_revised.pdf', bbox_inches='tight')
    plt.close()

create_gat_encoder_detail_figure_revised()