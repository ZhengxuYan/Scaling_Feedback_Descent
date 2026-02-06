import argparse
import logging
import os
from dotenv import load_dotenv

from utils.logging_setup import setup_logging, get_verbosity
from utils.file_io import load_json_file
from utils.api import APIClient
from core.elo import run_elo_analysis_creative

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run ELO Analysis Only for Creative Writing Benchmark.")
    parser.add_argument("--test-model", required=True, help="The model name or identifier for the test model.")
    parser.add_argument("--judge-model", required=True, help="The model name or identifier for the judge model.")
    parser.add_argument("--runs-file", default="creative_bench_runs.json", help="File where run data is stored.")
    parser.add_argument("--run-id", required=True, help="The specific run ID (or prefix) to analyze.")
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default="INFO")
    
    # Paths for necessary data files
    parser.add_argument("--creative-prompts-file", default="data/creative_writing_prompts_v3.json")
    parser.add_argument("--criteria-file", default="data/creative_writing_criteria.txt")
    parser.add_argument("--negative-criteria-file", default="data/negative_criteria.txt")
    parser.add_argument("--pairwise-prompt-file", default="data/pairwise_prompt.txt")
    parser.add_argument("--threads", type=int, default=10, help="Concurrency for ELO matchups.")

    args = parser.parse_args()
    setup_logging(get_verbosity(args.verbosity))

    # Construct the full run_key same way benchmark.py does
    # Note: benchmark.py uses f"{base_id}__{sanitized_model}"
    # But usually run-id passed in might be the base_id OR the full key. 
    # Let's check if the user provided the full key or just the prefix.
    # The user example was: --run-id "gemini-2.5-flash_235b_5"
    # In benchmark.py: base_id = run_id if run_id else str(uuid.uuid4()); run_key = f"{base_id}__{sanitized_model}"
    # If the user provides the EXACT run key in the JSON, we should try to find it.
    
    # Load runs file to verify/find the key
    if not os.path.exists(args.runs_file):
        logging.error(f"Runs file not found: {args.runs_file}")
        return

    runs_data = load_json_file(args.runs_file)
    run_key = args.run_id
    
    # Try direct match
    if run_key not in runs_data:
        # Try to construct it like benchmark.py does to see if it matches
        import re
        def sanitize_model_name(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
        
        sanitized_model = sanitize_model_name(args.test_model)
        possible_key = f"{run_key}__{sanitized_model}"
        
        if possible_key in runs_data:
            logging.info(f"Found run key '{possible_key}' matching prefix '{run_key}'")
            run_key = possible_key
        else:
             # Search for any key starting with run_id
            matches = [k for k in runs_data.keys() if k.startswith(run_key)]
            if len(matches) == 1:
                logging.info(f"Found run key '{matches[0]}' matching prefix '{run_key}'")
                run_key = matches[0]
            elif len(matches) > 1:
                logging.error(f"Multiple keys match prefix '{run_key}': {matches}. Please be more specific.")
                return
            else:
                logging.error(f"Run key '{run_key}' (or variants) not found in {args.runs_file}.")
                return

    logging.info(f"Using run key: {run_key}")

    # Load prompts
    if not os.path.exists(args.creative_prompts_file):
        logging.error(f"Creative prompts file not found: {args.creative_prompts_file}")
        return
    creative_prompts = load_json_file(args.creative_prompts_file)

    # Load negative criteria
    negative_criteria = []
    if os.path.exists(args.negative_criteria_file):
        with open(args.negative_criteria_file, "r", encoding="utf-8") as f:
            negative_criteria = [line.strip() for line in f if line.strip()]

    # Init Judge API Client
    # ELO only needs 'judge' client usually, unless it generates new text (which it shouldn't, only matchups)
    # The 'run_elo_analysis_creative' function signature expects 'api_clients' dict.
    logging.info("Initializing Judge API client...")
    api_clients = {
        "judge": APIClient(model_type="judge")
    }

    # Run ELO
    logging.info("Starting ELO Analysis...")
    run_elo_analysis_creative(
        run_key=run_key,
        elo_results_file="elo_results.json",
        test_model=args.test_model,
        judge_model=args.judge_model,
        api_clients=api_clients,
        writing_prompts=creative_prompts,
        concurrency=args.threads,
        pairwise_prompt_file=args.pairwise_prompt_file,
        negative_criteria=negative_criteria,
        creative_bench_runs_file=args.runs_file,
    )

    logging.info("ELO Analysis complete.")

if __name__ == "__main__":
    main()
