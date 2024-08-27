import json
import multiprocessing
import torch
import os
import argparse
import gzip
import logging
from functools import partial
from multiprocessing import Pool, Lock
from safetensors.torch import save_file, load_file
from tqdm import tqdm  # Import tqdm for progress bars


def process_file(input_file_path: str, output_directory: str, overwrite: bool):
    # Ensure the input file exists
    if not os.path.exists(input_file_path):
        logging.error(f"Input file '{input_file_path}' not found.")
        raise FileNotFoundError(f"Input file '{input_file_path}' not found.")

    # Get the base name of the input file for output file names
    base_name = (
        os.path.basename(input_file_path)
        .replace(".jsonl", "")
        .replace(".gz", "")
        .replace(".json", "")
    )

    vectors_path = os.path.join(output_directory, f"{base_name}_vectors.safetensors")
    docids_path = os.path.join(output_directory, f"{base_name}_docids.safetensors")

    if not overwrite:
        if os.path.exists(vectors_path) or os.path.exists(docids_path):
            error_msg = f"Output files '{vectors_path}' or '{docids_path}' already exist. Use '--overwrite' to overwrite."
            logging.error(error_msg)
            raise FileExistsError(error_msg)

    # Initialize lists to hold data
    vectors = []
    docids = []

    # Determine file opener based on file extension
    if input_file_path.endswith(".gz"):
        file_opener = gzip.open
    elif input_file_path.endswith(".jsonl") or input_file_path.endswith(".json"):
        file_opener = open
    else:
        logging.error("Input file must be a .json, .jsonl, or .gz file")
        raise ValueError("Input file must be a .json, .jsonl, or .gz file")

    # Get total number of lines for tqdm if possible
    try:
        total_lines = sum(1 for _ in file_opener(input_file_path, "rt"))
    except Exception:
        total_lines = None

    # Process the JSON, JSONL, or GZ file to extract vectors and docids
    try:
        with file_opener(input_file_path, "rt") as file:
            with tqdm(
                total=total_lines,
                desc=f"Processing {input_file_path}",
                position=multiprocessing.current_process()._identity[0],
                leave=True,
            ) as pbar:
                for line in file:
                    try:
                        entry = json.loads(line)
                        if isinstance(entry.get("vector", [None])[0], float):
                            vectors.append(entry["vector"])
                            docid = entry["docid"]
                            # Convert docid to ASCII values
                            docid_ascii = [ord(char) for char in docid]
                            docids.append(docid_ascii)
                        else:
                            logging.warning(
                                f"Skipped invalid vector entry with docid: {entry.get('docid', 'N/A')}"
                            )
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"Skipped invalid JSON entry: {e}")
                    finally:
                        pbar.update(1)
    except IOError as e:
        error_msg = f"Failed to read the input file '{input_file_path}': {e}"
        logging.error(error_msg)
        raise IOError(error_msg)

    # Convert lists to tensors
    vectors_tensor = torch.tensor(vectors, dtype=torch.float64)
    docids_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(d, dtype=torch.int64) for d in docids], batch_first=True
    )

    # Save the tensors to SafeTensors files
    try:
        save_file({"vectors": vectors_tensor}, vectors_path)
        save_file({"docids": docids_tensor}, docids_path)
        logging.info(f"Saved vectors to {vectors_path}")
        logging.info(f"Saved docids to {docids_path}")
    except IOError as e:
        error_msg = f"Failed to save tensors for {input_file_path}: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)

    # Load vectors and docids for verification
    try:
        loaded_vectors = load_file(vectors_path)["vectors"]
        loaded_docids = load_file(docids_path)["docids"]
        logging.info(f"Loaded vectors from {vectors_path}")
        logging.info(f"Loaded document IDs (ASCII) from {docids_path}")
        # Log detailed information to the file
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)
        logging.debug(f"Loaded vectors: {loaded_vectors}")
        logging.debug(f"Loaded document IDs (ASCII): {loaded_docids}")
    except IOError as e:
        error_msg = f"Failed to load tensors: {e}"
        logging.error(error_msg)
        raise IOError(error_msg)


if __name__ == "__main__":
    # Set up logging to both console and file
    log_file_path = "process_log.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # Log detailed information to a file
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),  # Log high-level information to the console
        ],
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process vectors and docids from JSON, JSONL, or GZ files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON, JSONL, or GZ directory",
    )
    parser.add_argument("--output", required=True, help="Path to the output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist",
    )

    args = parser.parse_args()

    # Define paths
    input_file_paths = [
        os.path.join(args.input, input_file_path)
        for input_file_path in os.listdir(args.input)
    ]
    output_directory = args.output

    # Ensure the output directory exists or create it
    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        error_msg = f"Failed to create output directory '{output_directory}': {e}"
        logging.error(error_msg)
        raise OSError(error_msg)

    # Set a lock for the tqdm bars
    tqdm.set_lock(Lock())

    num_processes = min(multiprocessing.cpu_count(), len(input_file_paths))

    # Use Pool to handle each file concurrently with the appropriate number of processes
    with Pool(processes=num_processes) as pool:
        process_file_fn = partial(
            process_file, output_directory=output_directory, overwrite=args.overwrite
        )
        pool.map(process_file_fn, input_file_paths)

    # Properly close the pool and release resources
    pool.join()
    pool.close()
