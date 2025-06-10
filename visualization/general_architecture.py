import matplotlib.pyplot as plt

def draw_box(ax, center, text, width=3, height=1.2, color="lightblue"):
    x, y = center
    rect = plt.Rectangle((x - width/2, y - height/2), width, height,
                         linewidth=1.5, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10)

def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.5))

fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure size

# Define vertical positions for boxes
positions = {
    "input": 12,
    "norm": 10,
    "encoder": 8,
    "embeddings": 6,
    "decoder": 4,
    "mlp": 4,
    "loss": 2
}

# Input box
draw_box(ax, (6, positions["input"]), "Input Graph\n(Node + Edge Features)")

# LayerNorm box
draw_box(ax, (6, positions["norm"]), "LayerNorm\n(Optional)")

# Encoder box
draw_box(ax, (6, positions["encoder"]), "Dynamic GNN Encoder")

# Embeddings box
draw_box(ax, (6, positions["embeddings"]), "Node Embeddings")

# Decoder (left side)
draw_box(ax, (2.5, positions["decoder"]), "Node Decoder")

# MLP (right side)
draw_box(ax, (9.5, positions["mlp"]), "Node Anomaly MLP")

# Loss function
draw_box(ax, (6, positions["loss"]), "Reconstruction Loss")

# Arrows for forward path
draw_arrow(ax, (6, positions["input"] - 0.6), (6, positions["norm"] + 0.6))
draw_arrow(ax, (6, positions["norm"] - 0.6), (6, positions["encoder"] + 0.6))
draw_arrow(ax, (6, positions["encoder"] - 0.6), (6, positions["embeddings"] + 0.6))

# From embeddings to decoder and MLP
draw_arrow(ax, (6, positions["embeddings"] - 0.6), (2.5, positions["decoder"] + 0.6))
draw_arrow(ax, (6, positions["embeddings"] - 0.6), (9.5, positions["mlp"] + 0.6))

# From decoder to loss
draw_arrow(ax, (2.5, positions["decoder"] - 0.6), (6, positions["loss"] + 0.6))

# Aesthetics
ax.set_xlim(0, 12)
ax.set_ylim(0, 14)
ax.axis('off')

plt.tight_layout()
plt.show()