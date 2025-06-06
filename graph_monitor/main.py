import argparse
import logging
import os
import time
from logging.handlers import RotatingFileHandler

# Import the workflow functions
# Ensure 'workflow' is a valid Python module in your environment
# For example, if monitor_new_files is in workflow.py, this import is correct.
try:
    from workflow import monitor_new_files
except ImportError:
    print("Error: Could not import 'monitor_new_files' from 'workflow.py'.")
    print("Please ensure 'workflow.py' is in the same directory or accessible via PYTHONPATH.")


    # Provide a dummy function or exit if workflow is critical
    def monitor_new_files(*args, **kwargs):
        logging.error("Workflow function not loaded. Cannot run monitoring.")
        raise SystemExit("Workflow module not found.")

# Define default paths here, can be overridden by arguments
DEFAULT_MODEL_SAVE_PATH = "model_checkpoints"
DEFAULT_STATS_SAVE_PATH = "stats"
DEFAULT_ANOMALY_LOG_PATH = "anomaly_logs"
DEFAULT_LOG_PATH = "logs"  # A default log path can be helpful if not specified


def setup_logging(args):
    """
    Configures the application's logging.

    Args:
        args: An argparse.Namespace object containing logging configuration.
    """
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times if this function is called repeatedly
    # This is crucial for long-running applications or re-initializing logging.
    # Check if any handlers are already configured.
    if not logger.handlers:
        # Define the log format
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Decide whether to log to console
        # Log to console if --log_file is NOT set (default behavior)
        # OR if --log_console_file IS set (explicitly request console + file)
        if not args.log_file or args.log_console_file:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            logger.addHandler(console_handler)
            logger.debug("Console handler added.")  # Using logger directly

        # Decide whether to log to file
        file_logging_enabled = False
        if args.log_file or args.log_console_file:
            if args.log_path:
                # Ensure log directory exists
                os.makedirs(args.log_path, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_filename = os.path.join(args.log_path, f"app_{timestamp}.log")

                file_handler = RotatingFileHandler(
                    log_filename,
                    maxBytes=args.log_max_bytes,
                    backupCount=args.log_backup_count
                )
                file_handler.setFormatter(log_format)
                logger.addHandler(file_handler)
                file_logging_enabled = True
                logger.info(f"Application logging to file: {log_filename}")
                logger.debug("File handler added.")
            else:
                logger.warning(
                    "File logging was requested (--log_file or --log_console_file enabled) "
                    "but --log_path was not specified. Logging to file will be disabled."
                )

        if not file_logging_enabled and (args.log_file or args.log_console_file):
            # This handles cases where file logging was requested but failed (e.g., no log_path),
            # and only console logging (if it was enabled) will be active.
            logger.warning(
                "File logging requested but could not be set up. Only console logging (if enabled) is active.")

    else:
        logger.debug("Logger handlers already configured. Skipping re-configuration.")

    # Avoid matplotlib font manager verbose logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--log_file",
        action='store_true',
        help="Enable logging to a file only. Requires --log_path. If not specified, logs to console by default."
    )
    group.add_argument(
        "--log_console_file",
        action='store_true',
        help="Enable logging to both console and a file. Requires --log_path."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=DEFAULT_LOG_PATH,  # Set a default log path
        help=f"Path to save the application log file (default: {DEFAULT_LOG_PATH}). "
             "Required if --log_file or --log_console_file is enabled and you want to customize it."
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
    parser.add_argument(
        "--visualization_path",
        type=str,
        default=None,
        help="Optional path to export node/edge visualizations and embeddings."
    )
    parser.add_argument(
        "--update_interval_seconds",
        type=int,
        default=30,
        help="Interval (in seconds) between graph updates and online learning steps. Default is 30."
    )
    parser.add_argument(
        "--export_period_updates",
        type=int,
        default=30,
        help="Number of update intervals for exporting visualization. Default is 30"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "detect"],
        help="Specify the mode: 'train' for learning normal patterns, 'detect' for anomaly recognition.",
    )
    args = parser.parse_args()

    # --- Setup Logging First ---
    setup_logging(args)

    # --- Application Logic ---
    mode = args.mode
    logging.info(f"Running in '{mode}' mode.")

    target_directory = args.path
    model_save_path = args.model_path
    stats_save_path = args.stats_path
    anomaly_log_path = args.anomaly_path

    # Create necessary directories
    os.makedirs(anomaly_log_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(stats_save_path, exist_ok=True)
    if args.visualization_path:
        os.makedirs(args.visualization_path, exist_ok=True)

    logging.info(f"Monitoring directory: {target_directory}")
    logging.info(f"Model will be saved to: {model_save_path}")
    logging.info(f"Statistics will be saved to: {stats_save_path}")
    logging.info(f"Anomaly logs will be saved to: {anomaly_log_path}")

    train_bool = (mode == 'train')  # More concise way to set boolean

    try:
        monitor_new_files(
            target_directory,
            model_save_path,
            stats_save_path,
            anomaly_log_path,
            update_interval_seconds=args.update_interval_seconds,
            export_period_updates=args.export_period_updates,
            visualization_path=args.visualization_path,
            train_mode=train_bool
        )
    except Exception as e:
        logging.critical(f"Application terminated due to an unhandled error: {e}", exc_info=True)