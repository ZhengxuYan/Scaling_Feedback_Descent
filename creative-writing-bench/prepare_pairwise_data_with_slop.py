
import json
import os
import random
import sys
import statistics
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict

# Add parent directory to path to import text_metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_metrics import compute_metrics

# Metrics where lower is better
NEGATIVE_METRICS = [
    'Meandering', 'Weak Dialogue', "Tell-Don't-Show", 'Unsurprising or Uncreative',
    'Amateurish', 'Purple Prose', 'Overwrought', 'Incongruent Ending Positivity',
    'Unearned Transformations',
    'Slop Score' # Added Slop Score
]

def load_run_data(filepath: str) -> Dict[str, Any]:
    print(f"Loading run data from {filepath}...")
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_aggregated_score(scores: Dict[str, float]) -> float:
    """
    Calculates the aggregated score by averaging all JUDGE metrics.
    Negative metrics are inverted (20 - score).
    """
    total_score = 0.0
    count = 0
    
    for metric, val in scores.items():
        if metric == "Overall Impression": # Skip Overall Impression in the average
            continue
        
        # Skip Slop Score for the base aggregated score calculation
        if metric == "Slop Score":
            continue
            
        score = float(val)
        if metric in NEGATIVE_METRICS:
            score = 20.0 - score
            
        total_score += score
        count += 1
        
    if count == 0:
        return 0.0
        
    return total_score / count

def extract_candidates(run_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Groups candidates by prompt_id.
    """
    grouped_data = {}
    
    print("Extracting candidates...")
    for run_key, run_info in run_data.items():
        test_model = run_info.get("test_model", "unknown")
        creative_tasks = run_info.get("creative_tasks", {})
        
        # Iterate through iterations (1, 2, 3...)
        for iter_key, prompts_dict in creative_tasks.items():
            for prompt_id, task_info in prompts_dict.items():
                
                # We only care about completed tasks with scores
                if task_info.get("status") not in ["completed", "judged"]:
                    continue
                
                base_prompt = task_info.get("base_prompt", "")
                results_by_mod = task_info.get("results_by_modifier", {})
                
                # There might be multiple modifiers (seeds), usually 1 per iteration
                for mod_key, res_block in results_by_mod.items():
                    judge_scores = res_block.get("judge_scores", {})
                    model_response = res_block.get("model_response", "")
                    
                    if not judge_scores or not model_response:
                        continue
                        
                    # Get Overall Impression score (still useful for reference)
                    overall_score = judge_scores.get("Overall Impression", 0)
                    
                    # Calculate Aggregated Score (Judge only)
                    aggregated_score = calculate_aggregated_score(judge_scores)
                    
                    # Calculate Slop Score
                    try:
                        metrics_res = compute_metrics(model_response, metrics=["slop_score"])
                        slop_score = metrics_res["slop_score"]
                    except Exception as e:
                        print(f"Error computing slop score: {e}")
                        slop_score = 0.0
                    
                    if prompt_id not in grouped_data:
                        grouped_data[prompt_id] = {
                            "base_prompt": base_prompt,
                            "candidates": []
                        }
                    
                    # Add Slop Score to scoring dict for rationale generation
                    combined_scores = judge_scores.copy()
                    combined_scores['Slop Score'] = slop_score
                    
                    grouped_data[prompt_id]["candidates"].append({
                        "model_name": test_model,
                        "summary": model_response,
                        "scores": combined_scores,
                        "overall_score": overall_score,
                        "aggregated_score": aggregated_score,
                        "slop_score": slop_score
                    })
                    
    print(f"Found {len(grouped_data)} prompts with candidates.")
    return grouped_data

def normalize_scores_and_compute_final(grouped_data: Dict[str, Dict[str, Any]]):
    """
    Normalizes aggregated_score and slop_score using Z-score (standardization)
    and computes the final score.
    """
    print("Normalizing scores...")
    
    all_agg_scores = []
    all_slop_scores = []
    
    # First pass: Collect all scores
    for prompt_id, data in grouped_data.items():
        for cand in data['candidates']:
            all_agg_scores.append(cand['aggregated_score'])
            all_slop_scores.append(cand['slop_score'])
            
    if not all_agg_scores:
        return
        
    mean_agg = statistics.mean(all_agg_scores)
    std_agg = statistics.stdev(all_agg_scores) if len(all_agg_scores) > 1 else 1.0
    if std_agg == 0: std_agg = 1.0

    mean_slop = statistics.mean(all_slop_scores)
    std_slop = statistics.stdev(all_slop_scores) if len(all_slop_scores) > 1 else 1.0
    if std_slop == 0: std_slop = 1.0
    
    print(f"Aggregated Score: mean={mean_agg:.2f}, std={std_agg:.2f}")
    print(f"Slop Score: mean={mean_slop:.2f}, std={std_slop:.2f}")
    
    # Second pass: Compute final score
    for prompt_id, data in grouped_data.items():
        for cand in data['candidates']:
            raw_agg = cand['aggregated_score']
            raw_slop = cand['slop_score']
            
            z_agg = (raw_agg - mean_agg) / std_agg
            z_slop = (raw_slop - mean_slop) / std_slop
            
            # Final Score: Higher Agg is good, Lower Slop is good.
            # We want to maximize Final Score.
            # final = z_agg - z_slop 
            # (or average of z_agg and -z_slop -> same ranking order)
            
            final_score = z_agg - z_slop
            cand['final_score'] = final_score
            cand['z_agg'] = z_agg
            cand['z_slop'] = z_slop

def create_synthetic_rationale(winner_cand, loser_cand):
    """
    Synthesize a rationale based on score differences.
    """
    w_scores = winner_cand['scores']
    l_scores = loser_cand['scores']
    
    details = []
    
    # Metrics where higher is better
    positive_metrics = [
        'Adherence to Instructions', 'Believable Character Actions', 'Nuanced Characters',
        'Consistent Voice/Tone of Writing', 'Imagery and Descriptive Quality', 'Elegant Prose',
        'Emotionally Engaging', 'Emotionally Complex', 'Coherent', 'Well-earned Lightness or Darkness',
        'Sentences Flow Naturally', 'Overall Reader Engagement'
    ]
    
    diffs = []
    
    # Check positive metrics (Winner - Loser)
    for m in positive_metrics:
        w_val = float(w_scores.get(m, 0))
        l_val = float(l_scores.get(m, 0))
        diff = w_val - l_val
        
        if diff != 0:
            if diff > 0:
                diffs.append((diff, f"better {m}", w_val, l_val))
            else:
                diffs.append((diff, f"worse {m}", w_val, l_val))
            
    # Check negative metrics (Loser - Winner) -> Winner is better because it has LESS of the negative trait
    for m in NEGATIVE_METRICS:
        w_val = float(w_scores.get(m, 0))
        l_val = float(l_scores.get(m, 0))
        diff = l_val - w_val # Positive diff means Winner is lower (better)
        
        if diff != 0:
            if diff > 0:
                diffs.append((diff, f"less {m}", w_val, l_val))
            else:
                diffs.append((diff, f"more {m}", w_val, l_val))
            
    # Sort by magnitude of difference descending (benefit to winner)
    # We want to show the biggest "pros" first.
    diffs.sort(key=lambda x: x[0], reverse=True)
    
    for _, reason, w_val, l_val in diffs:
        # Format the values nicely
        w_fmt = f"{w_val:.2f}"
        l_fmt = f"{l_val:.2f}"
        details.append(f"{reason} ({w_fmt} vs {l_fmt})")
            
    reason_text = ", ".join(details) if details else "better overall quality"
    
    return f"The winner produced a better draft because it has {reason_text}. (Final Score (Z-normalized diff): {winner_cand['final_score']:.2f} vs {loser_cand['final_score']:.2f})"

def construct_comparison_data():
    runs_file = "creative_bench_runs.json"
    if not os.path.exists(runs_file):
        print(f"Error: {runs_file} not found.")
        return

    run_data = load_run_data(runs_file)
    grouped_data = extract_candidates(run_data)
    
    # Normalize scores and compute final_score
    normalize_scores_and_compute_final(grouped_data)
    
    training_examples = []
    
    print("Creating pairs...")
    for prompt_id, data in grouped_data.items():
        cands = data['candidates']
        prompt_pairs = []
        
        if len(cands) < 2:
            continue
            
        # Generate all pairs
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                cand_a = cands[i]
                cand_b = cands[j]
                
                # Use FINAL SCORE for comparison
                score_a = cand_a.get('final_score', 0)
                score_b = cand_b.get('final_score', 0)
                
                # Skip if scores are too close
                # Since we are using Z-scores, "0.5" might be a bit large?
                # Z-score of 0.5 is half a standard deviation. That's a reasonable mandatory gap.
                if abs(score_a - score_b) < 0.5:
                    continue
                
                # Randomize order to avoid position bias
                if random.random() < 0.5:
                    cand_a, cand_b = cand_b, cand_a
                    score_a, score_b = score_b, score_a
                
                if score_a > score_b:
                    ground_truth_winner = "Draft A"
                    winner_cand = cand_a
                    loser_cand = cand_b
                else:
                    ground_truth_winner = "Draft B"
                    winner_cand = cand_b
                    loser_cand = cand_a
                
                rationale = create_synthetic_rationale(winner_cand, loser_cand)
                
                user_content = (
                    f"Here is a writing prompt:\n{data['base_prompt']}\n\n"
                    f"Draft A:\n{cand_a['summary']}\n\n"
                    f"Draft B:\n{cand_b['summary']}\n\n"
                    "Which draft is better and why?"
                )
                
                assistant_content = f"**Verdict**: {ground_truth_winner}\n\n**Feedback**: {rationale}"
                
                prompt_pairs.append({
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                })
        
        # Downsample if too many pairs for this prompt
        if len(prompt_pairs) > 500:
            print(f"  Downsampling prompt {prompt_id} from {len(prompt_pairs)} to 500 pairs.")
            prompt_pairs = random.sample(prompt_pairs, 500)
            
        training_examples.extend(prompt_pairs)

    print(f"Created {len(training_examples)} comparison pairs.")
    
    # Shuffle
    random.shuffle(training_examples)

    # Split into train and test (90/10 split)
    total_samples = len(training_examples)
    test_size = max(500, int(total_samples * 0.1)) 
    
    if total_samples > test_size:
        test_data = training_examples[:test_size]
        train_data = training_examples[test_size:]
    else:
        print("Warning: Not enough data for split. Using all for train.")
        train_data = training_examples
        test_data = []

    os.makedirs("data", exist_ok=True)
    
    # Save locally
    with open("data/pairwise_train_slop.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("data/pairwise_test_slop.jsonl", "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")

    print("Data preparation complete.")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    try:
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data)
        })
        
        repo_id = "JasonYan777/creative-writing-pairwise-critic-with-slop"
        dataset_dict.push_to_hub(repo_id)
        print(f"Successfully pushed to {repo_id}")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")

if __name__ == "__main__":
    construct_comparison_data()
