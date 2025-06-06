# Network Anomaly Detection System

This repository contains a system for real-time anomaly detection in network graphs using a Graph Neural Network (GNN) based autoencoder. The system monitors a specified directory for new network graph files (in `.dot` format), processes them, and identifies anomalous nodes. It can operate in a training mode (for initial model training and online updates) or a detection-only mode.

---

## Features

* **Real-time Monitoring**: Continuously scans a directory for new `.dot` graph files.
* **GNN-based Anomaly Detection**: Utilizes a Graph Attention Network (GAT) based autoencoder to learn normal network behavior and detect deviations.
* **Online Learning**: Supports online updates to the GNN model, allowing it to adapt to evolving network patterns (in training mode).
* **Reconstruction and MLP-based Anomaly Scoring**: Identifies anomalies based on reconstruction errors and a subsequent MLP-driven anomaly score.
* **Configurable Parameters**: Adjustable parameters for monitoring interval, training epochs, update steps, and visualization frequency.
* **Detailed Logging**: Comprehensive logging of system operations, anomaly detections, and errors.
* **Visualization**: Generates 3D t-SNE plots of node embeddings, highlighting anomalous nodes, and 2D visualizations of node features.
* **Anomaly Logging**: Saves detected anomalies in a structured JSON format for further analysis.

---

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/network-anomaly-detection.git
    cd network-anomaly-detection
    ```

2. **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` contains `torch`, `torch_geometric`, `networkx`, `scikit-learn`, `matplotlib`, `numpy`, `loguru` (if used), etc.)*

---

## Usage

The system can be run using the `main.py` script.

### Command-line Arguments

* `--path <directory>`: **(Required)** Path to the directory to monitor for `.dot` graph files.
* `--model_path <file>`: Path to save/load the GNN model checkpoint. *(Default: `model_checkpoint.pt`)*
* `--stats_path <file>`: Path to save/load the running statistics (mean/std for normalization). *(Default: `running_stats.pkl`)*
* `--log_path <directory>`: Path to the directory for saving log files. *(Default: `logs`)*
* `--anomaly_log_path <directory>`: Path to the directory for saving anomaly detection logs. *(Default: `anomaly_logs`)*
* `--update_interval_seconds <seconds>`: How often (in seconds) to check the directory for new files. *(Default: `30`)*
* `--export_period_updates <num_updates>`: Frequency (in number of processed updates) to export visualizations. *(Default: `50`)*
* `--visualization_path <directory>`: Path to the directory for saving visualization images. If not provided, visualizations are skipped. *(Default: `None`)*
* `--train_mode`: If set, the system will perform initial training and continuous online updates. If not set, it operates in detection-only mode, loading a pre-trained model. *(Default: `False`)*
* `--log_console_file`: If set, logs will be printed to the console and saved to a file. If not set, logs are only saved to a file.
* `--log_level <level>`: Sets the logging level (e.g., `INFO`, `DEBUG`, `WARNING`, `ERROR`). *(Default: `INFO`)*

---

## Usage Examples

Here are a couple of examples demonstrating how to run the anomaly detection system from your command line:

### Basic Monitoring

This command starts monitoring the `/home/lu/Desktop/output/` directory for new `.dot` files. It will log output to both the console and a file within the `logs/` directory, showing `INFO` level messages and above.

```bash
python main.py --path /home/lu/Desktop/output/ --log_console_file --log_level INFO --log_path logs
```

### Detection Mode with Visualization

This example runs the system in detection mode (`--mode detect`), meaning it loads a pre-trained model and only performs anomaly detection, without continuous online training. It monitors the `/home/lu/Desktop/output_ssl_bruteforce/` directory, checks for new files every 60 seconds, and exports **visualization features** (like 3D embedding plots) to the `vis/` directory every 5 updates. Logging will be set to `DEBUG` level for more verbose output.

```bash
python main.py --path /home/lu/Desktop/output_ssl_bruteforce/ --log_console_file --log_level DEBUG --log_path logs --visualization_path vis --update_interval_seconds 60 --export_period_updates 5 --mode detect
```

---

## Output Structure

### Anomaly Logs (`anomaly_logs/`)

Anomaly detection results are saved in JSON files, providing a detailed record of unusual activity. Each file corresponds to a specific update and contains the following structure:

- **`timestamp`**: The UTC timestamp (e.g., `"YYYYMMDD_HHMMSS_UTC"`) indicating when the anomalies were detected.
- **`update_count`**: A sequential integer representing the number of data files processed up to this point.
- **`nodes_in_graph`**: The total number of nodes present in the graph.
- **`node_anomalies`**: A list of detected anomalous nodes. Each entry includes:
  - `node_id`: The internal index of the anomalous node.
  - `detected_by`: Method used for detection (e.g., `"reconstruction"`, `"mlp"`).
  - `recon_error`: Reconstruction error value.
  - `mlp_score`: Anomaly score from the MLP.
  - `ip`: IP address of the node (if available).

**Example Log Entry**:
```json
{
    "timestamp": "20250606_173542",
    "update_count": 1,
    "nodes_in_graph": 20,
    "node_anomalies": [
        {
            "node_id": 8,
            "detected_by": "reconstruction",
            "recon_error": 0.706444263458252,
            "mlp_score": 1.0833422940356199e-31,
            "ip": "192.168.27.10"
        }
    ]
}
```

### Visualizations (`vis/`)

If `--visualization_path` is specified, the system generates various plots to help understand the network and detected anomalies:

- **3D Embedding Plots**: t-SNE-based plots that reduce node embeddings to 3D for visualization. Anomalies are highlighted (e.g., in red). Filenames: `gcn_tsne_anomalies_<update_count>.png`
- **Node Feature Visualizations**: 2D histograms or scatter plots showing distributions of node features.

---

## Project Structure (Assumed)

```
.
├── main.py                     # Main script to run the monitoring system
├── workflow.py                 # Contains the monitor_new_files function and overall logic
├── model.py                    # Defines NodeGNNAnomalyDetector, Autoencoder, etc.
├── utils.py                    # Utility functions (load/save checkpoint, stats, logging setup)
├── processing.py               # Handles .dot file parsing and graph data preparation
├── visualization.py            # Contains visualization functions (e.g., visualize_embeddings_3d)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements.

---

## License

(Add your license information here, e.g., MIT License)
