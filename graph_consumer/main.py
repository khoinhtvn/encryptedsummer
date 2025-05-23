# your_main_script.py
import argparse
import logging
import os
import time
from logging.handlers import RotatingFileHandler

# Import the workflow functions
from workflow import process_and_learn # Import the function

# Define default paths here, can be overridden by arguments
DEFAULT_MODEL_SAVE_PATH = "model_checkpoints"
DEFAULT_STATS_SAVE_PATH = "stats"
DEFAULT_ANOMALY_LOG_PATH = "anomaly_logs"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a directory for new or updated .dot graph files. "
            "For each change, the graph structure is updated and used to incrementally train "
            "a neural network model in an online fashion."
        )
    )
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the directory to monitor (e.g., samples/big_web_enum)"
    )
    parser.add_argument(
        "--log_level", "-l",
        type=str,
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to save the application log file. If not specified, logs will only be printed to the console."
    )
    parser.add_argument(
        "--log_max_bytes",
        type=int,
        default=10 * 1024 * 1024,  # 10 MB
        help="Maximum size of the application log file before it rolls over (in bytes)."
    )
    parser.add_argument(
        "--log_backup_count",
        type=int,
        default=5,
        help="Number of backup application log files to keep."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_SAVE_PATH,
        help=f"Path to save the model checkpoints (default: {DEFAULT_MODEL_SAVE_PATH})"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default=DEFAULT_STATS_SAVE_PATH,
        help=f"Path to save training statistics (default: {DEFAULT_STATS_SAVE_PATH})"
    )
    parser.add_argument(
        "--anomaly_path",
        type=str,
        default=DEFAULT_ANOMALY_LOG_PATH,
        help=f"Path to save anomaly logs (default: {DEFAULT_ANOMALY_LOG_PATH})"
    )

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    target_directory = args.path
    model_save_path = args.model_path
    stats_save_path = args.stats_path
    anomaly_log_path = args.anomaly_path

    # Create directories if they don't exist
    os.makedirs(anomaly_log_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(stats_save_path, exist_ok=True)

    # Configure logging (same as before)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    if args.log_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(args.log_path, f"app_{timestamp}.log")
        os.makedirs(args.log_path, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=args.log_max_bytes,
            backupCount=args.log_backup_count
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.info(f"Application logging to file: {log_filename}")

    if not os.path.isdir(target_directory):
        logging.error(f"Error: Directory '{target_directory}' does not exist.")
    else:
        logging.info(f"Starting to monitor directory: {target_directory} with log level: {args.log_level}")
        # Call the process_and_learn function with the paths
        process_and_learn(target_directory, model_save_path, stats_save_path, anomaly_log_path, update_interval_seconds=30)