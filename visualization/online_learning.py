import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

def create_online_learning_figure_revised():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    online_learning_color = "#4caf50" # Light Green

    def draw_process_box(center_x, center_y, width, height, title, icon, details, color='white'):
        rect = patches.FancyBboxPatch((center_x - width/2, center_y - height/2),
                                      width, height,
                                      boxstyle="round,pad=0.4",
                                      facecolor=color, edgecolor=online_learning_color, linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(center_x, center_y + height/2 * 0.5, icon, ha='center', va='center', fontsize=25, color=online_learning_color)
        ax.text(center_x, center_y + height/2 * 0.1, title, ha='center', va='center', fontsize=12, weight='bold', color='black')
        ax.text(center_x, center_y - height/2 * 0.2, details, ha='center', va='top', fontsize=9, color='dimgray', linespacing=1.2)


    # Main Title Box
    main_box_y = 5.5
    main_box_height = 0.8
    rect = patches.FancyBboxPatch((5 - 4/2, main_box_y - main_box_height/2),
                                  4, main_box_height,
                                  boxstyle="round,pad=0.6",
                                  facecolor=online_learning_color, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(5, main_box_y, "Online Learning & Adaptation Framework", ha='center', va='center', fontsize=14, weight='bold', color='white')

    # Components
    component_y = 3
    component_width = 2.8
    component_height = 2.5

    draw_process_box(2, component_y, component_width, component_height,
                     "Replay Buffer", "📚",
                     "• Stores 2000 representative graphs\n"
                     "• Random sampling for training\n"
                     "• Edge dropout augmentation")

    draw_process_box(5, component_y, component_width, component_height,
                     "Statistical Monitoring", "📊",
                     "• Feature statistics (Welford's)\n"
                     "• Reconstruction error tracking\n"
                     "• Drift detection")

    draw_process_box(8, component_y, component_width, component_height,
                     "Adaptive Learning", "⚡",
                     "• Cosine annealing scheduler\n"
                     "• Warm restarts\n"
                     "• Dynamic threshold adjustment")

    # Arrows indicating interaction/flow (conceptual, not strict data flow)
    arrow_props = dict(facecolor='grey', shrink=0.05, width=0.08, headwidth=0.3)
    ax.annotate("", xy=(3.5, 3.5), xytext=(4.5, 3.5), arrowprops=arrow_props) # Buffer to Monitoring
    ax.annotate("", xy=(6.5, 3.5), xytext=(5.5, 3.5), arrowprops=arrow_props) # Monitoring to Adaptive
    ax.annotate("", xy=(5, 4.5), xytext=(5, 5), arrowprops=arrow_props) # Output of components back to main frame

    ax.text(3.5, 3.7, "Data Feed", ha='center', va='bottom', fontsize=8, color='dimgray')
    ax.text(6.5, 3.7, "Feedback", ha='center', va='bottom', fontsize=8, color='dimgray')

    ax.set_title("Figure 4: Online Learning & Adaptation Framework", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('online_learning_framework_revised.pdf', bbox_inches='tight')
    plt.close()

create_online_learning_figure_revised()