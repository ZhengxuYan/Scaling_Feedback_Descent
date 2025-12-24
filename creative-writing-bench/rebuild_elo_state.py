
import json
import os
import logging
from collections import defaultdict
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RUNS_FILE = "creative-writing-bench/creative_bench_runs.json"
ELO_FILE = "creative-writing-bench/elo_results.json"
DEFAULT_ELO = 1200.0

def rebuild_elo_state():
    if not os.path.exists(RUNS_FILE):
        logging.error(f"Runs file not found: {RUNS_FILE}")
        return

    logging.info(f"Loading runs from {RUNS_FILE}...")
    with open(RUNS_FILE, 'r') as f:
        runs_data = json.load(f)

    # Initialize or load existing ELO data
    if os.path.exists(ELO_FILE):
        logging.info(f"Loading existing ELO data from {ELO_FILE}...")
        with open(ELO_FILE, 'r') as f:
            elo_data = json.load(f)
    else:
        logging.info("Initializing fresh ELO data structure.")
        elo_data = {"__metadata__": {}}

    models_processed = 0
    
    for run_key, run_info in runs_data.items():
        test_model = run_info.get("test_model")
        if not test_model:
            continue

        logging.info(f"Processing run for model: {test_model}")
        
        # Ensure model entry exists
        if test_model not in elo_data:
            elo_data[test_model] = {
                "elo": DEFAULT_ELO,
                "creative_writing_rubric_score_agg": 0.0,
                "iterations": {},
                "elo_analysis": {"pairwise_comparisons": []}
            }

        # 1. Update ELO if available in the run result
        # We trust the run's recorded ELO as the "last known good" state
        run_elo = run_info.get("results", {}).get("benchmark_results", {}).get("elo_raw")
        if run_elo is not None:
            try:
                elo_data[test_model]["elo"] = float(run_elo)
                logging.info(f"  - Set ELO to {run_elo}")
            except (ValueError, TypeError):
                logging.warning(f"  - Could not convert ELO '{run_elo}' to float. Keeping default/existing.")

        # 2. Update Rubric Score Agg
        run_rubric = run_info.get("results", {}).get("benchmark_results", {}).get("creative_score_0_20")
        if run_rubric is not None:
            elo_data[test_model]["creative_writing_rubric_score_agg"] = float(run_rubric)
            logging.info(f"  - Set Rubric Score to {run_rubric}")

        # 3. Extract Items and Iteration Scores
        # This mimics the aggregation logic in core/elo.py
        creative_tasks = run_info.get("creative_tasks", {})
        
        # We need to group by iteration ID (e.g., "1", "2")
        # Structure of creative_tasks: { "1": { "item_id": { ...task_info... } } }
        
        for iter_id, prompt_map in creative_tasks.items():
            if iter_id not in elo_data[test_model]["iterations"]:
                elo_data[test_model]["iterations"][iter_id] = {
                    "creative_writing_rubric_score_iter": 0.0, # Will calc below
                    "items": {},
                    "item_scores": {}
                }
            
            iter_entry = elo_data[test_model]["iterations"][iter_id]
            
            scores_accum = 0.0
            scores_count = 0
            
            for item_id, task_info in prompt_map.items():
                # Extract text
                # results_by_modifier -> seed -> model_response
                results_by_modifier = task_info.get("results_by_modifier", {})
                combined_text = ""
                item_score_sum = 0.0
                item_score_count = 0
                
                for _, block in results_by_modifier.items():
                    txt = block.get("model_response", "").strip()
                    if txt:
                        combined_text += txt + "\n"
                    
                    # Extract scores
                    judge_scores = block.get("judge_scores", {})
                    for metric, val in judge_scores.items():
                        if isinstance(val, (int, float)):
                            # Note: We are NOT doing the 'invert_if_negative' logic here strictly 
                            # because we don't have the negative_criteria list easily accessible without importing config.
                            # However, usually the 'creative_score_0_20' in benchmark_results is already the aggregated one.
                            # For 'item_scores', we might be slightly off if we don't invert, 
                            # BUT for the purpose of just having text to judge, the score is secondary metadata.
                            # Let's try to use the task_info's score if pre-calculated, but it usually isn't.
                            # We will just sum raw values for now as a proxy, or better yet, 
                            # rely on the fact that we just need the TEXT for future comparisons.
                            item_score_sum += val
                            item_score_count += 1

                if combined_text:
                    iter_entry["items"][item_id] = combined_text.strip()
                
                if item_score_count > 0:
                    avg_val = item_score_sum / item_score_count
                    iter_entry["item_scores"][item_id] = avg_val
                    scores_accum += avg_val
                    scores_count += 1
            
            # Update iteration average if we have data
            if scores_count > 0:
                iter_entry["creative_writing_rubric_score_iter"] = scores_accum / scores_count

        models_processed += 1

    logging.info(f"Rebuild complete. Processed {models_processed} models.")
    
    with open(ELO_FILE, 'w') as f:
        json.dump(elo_data, f, indent=2)
    logging.info(f"Saved ELO data to {ELO_FILE}")

if __name__ == "__main__":
    rebuild_elo_state()
