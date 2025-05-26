# Network Threat Graph Toolkit

This project provides a complete pipeline for network anomaly detection using Graph Neural Networks (GNNs). It is designed to transform raw network activity (via Zeek logs) into structured graph representations, learn from them using temporal GNNs, and detect anomalous behaviors over time.

---

## 🛠️ Components

### 1. **Graph Monitor**

* A Python module that consumes graph data from `.dot` files.
* Uses an adaptive GNN model to track and learn network behavior.
* Periodically visualizes and exports node/edge features and graph embeddings.
* Detects and logs anomalies based on structural and feature deviations.

### 2. **Log2Graph**

* A C++ utility that parses Zeek log files.
* Converts each time window of logs into a graph in DOT format.
* Defines nodes and edges based on network entities and interactions.

### 3. **Zeek Replay Utility (Docker)**

* Docker container for replaying PCAP files using Zeek.
* Automatically generates logs in a mounted volume.
* Useful for simulations, offline testing, or reproducing traffic behavior.

---

## 🚀 Quick Start

### Step 1: Launch the Python Graph Monitor

```bash
python3 main.py \
  -p <graph_output_directory> \
  --log_level DEBUG \
  --log_path logs \
  --visualization_path vis \
  --export_period_updates 10
```

* Monitors the `<graph_output_directory>` folder for `.dot` graph files.
* Logs GNN activity and anomaly detection results to `logs/`.
* Saves feature visualizations and embeddings every 10 updates to `vis/`.

### Step 2: Start the C++ Log2Graph Parser

```bash
./Log2Graph \
  <zeek_logs_directory> \
  --export-path <graph_output_directory>
```

* Parses Zeek logs in the specified directory.
* Converts them into `.dot` files for each time interval.
* Saves DOT graphs to the `<graph_output_directory>` folder.

### Step 3 (Optional): Start Docker PCAP Replay

```bash
docker run -it --rm \
  --name zeek-replayer \
  --privileged \
  -v <pcap_to_replay_path>:/pcap:ro \
  -v <output_zeek_path>:/zeek-logs:rw \
  tkhoi/zeek-replayer:latest
```

* Replays traffic from PCAP files in `<pcap_to_replay_path>`.
* Generates Zeek logs in `<output_zeek_path>`.

Alternatively, wait for live Zeek logging to populate the logs directory.

---

## 📂 Output Structure

### Graph Monitor

* `logs/`: GNN logs and anomaly events.
* `vis/`: Visualizations of node/edge features and embeddings.
* `output/`: Graph embeddings (if enabled).

### Log2Graph

* `output/`: One `.dot` graph file per time window.

---

## 📆 Requirements

### Python (Graph Monitor)

* Python 3.8+
* PyTorch
* NetworkX
* Matplotlib
* scikit-learn
* tqdm

Install via:

```bash
pip install -r requirements.txt
```

### C++ (Log2Graph)

* C++17 compliant compiler
* CMake 3.10+
* Boost (optional, if used for parsing or file I/O)

### Docker (Zeek Replay)

* Docker engine
* Internet access (to pull `tkhoi/zeek-replayer` image, also part of this repo)

---


## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🚫 Disclaimer

This project is a research prototype and should not be used in production without thorough validation. Always ensure appropriate permissions before capturing or replaying network traffic.

---

For questions or contributions, feel free to open an issue or submit a pull request!
