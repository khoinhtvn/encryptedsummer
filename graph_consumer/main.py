import os
import argparse
from workflow import process_and_learn

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

    args = parser.parse_args()
    target_directory = args.path

    if not os.path.isdir(target_directory):
        print(f"Error: Directory '{target_directory}' does not exist.")
    else:
        print(f"Starting to monitor directory: {target_directory}")
        process_and_learn(target_directory)
