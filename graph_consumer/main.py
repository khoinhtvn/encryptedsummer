import os
import argparse
from workflow import process_and_learn
import logging
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

    args = parser.parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    target_directory = args.path

    if not os.path.isdir(target_directory):
        logging.error(f"Error: Directory '{target_directory}' does not exist.")
    else:
        logging.info(f"Starting to monitor directory: {target_directory} with log level: {args.log_level}")
        process_and_learn(target_directory)