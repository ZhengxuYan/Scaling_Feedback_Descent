import json
import os
import random
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict

# Metrics where lower is better
NEGATIVE_METRICS = [
    'Meandering', 'Weak Dialogue', "Tell-Don't-Show", 'Unsurprising or Uncreative',
    'Amateurish', 'Purple Prose', 'Overwrought', 'Incongruent Ending Positivity',
    'Unearned Transformations'
]

def load_run_data(filepath: str) -> Dict[str, Any]:
    print(f"Loading run data from {filepath}...")
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_aggregated_score(scores: Dict[str, float]) -> float:
    """
    Calculates the aggregated score by averaging all metrics.
    Negative metrics are inverted (20 - score).
    """
    total_score = 0.0
    count = 0
    
    for metric, val in scores.items():
        if metric == "Overall Impression": # Skip Overall Impression in the average
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
    Returns:
        {
            prompt_id: {
                "base_prompt": str,
                "candidates": [
                    {
                        "model_name": str,
                        "summary": str, # The creative writing output
                        "scores": dict,
                        "overall_score": float,
                        "aggregated_score": float
                    }
                ]
            }
        }
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
                    
                    # Calculate Aggregated Score
                    aggregated_score = calculate_aggregated_score(judge_scores)
                    
                    if prompt_id not in grouped_data:
                        grouped_data[prompt_id] = {
                            "base_prompt": base_prompt,
                            "candidates": []
                        }
                    
                    grouped_data[prompt_id]["candidates"].append({
                        "model_name": test_model,
                        "summary": model_response,
                        "scores": judge_scores,
                        "overall_score": overall_score,
                        "aggregated_score": aggregated_score
                    })
                    
    print(f"Found {len(grouped_data)} prompts with candidates.")
    return grouped_data

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
        # We include all differences, even if 0 or negative (though winner usually has positive net diffs)
        # But to make the rationale readable "better X", we usually focus on where winner is better.
        # However, user asked for "all the score comparison". 
        # If winner is WORSE in some metric, we should probably still mention it for completeness?
        # "Summary A is better because it has better X... but worse Y..."
        # Let's stick to the "better X" / "less Y" format for now as it supports the verdict.
        # If we list *everything*, it might be confusing if the winner is worse on some.
        # But the prompt says "include all the score comparison".
        # Let's list where the winner is better first, then maybe where it's worse?
        # Or just list everything sorted by net benefit to winner.
        
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
    
    # top_diffs = diffs[:3] # Removed limit
    
    for _, reason, w_val, l_val in diffs:
        details.append(f"{reason} ({w_val} vs {l_val})")
            
    reason_text = ", ".join(details) if details else "better overall quality"
    
    return f"The winner produced a better story because it has {reason_text}. (Overall Score: {winner_cand['aggregated_score']:.2f} vs {loser_cand['aggregated_score']:.2f})"

def construct_comparison_data():
    runs_file = "creative_bench_runs.json"
    if not os.path.exists(runs_file):
        print(f"Error: {runs_file} not found.")
        return

    run_data = load_run_data(runs_file)
    grouped_data = extract_candidates(run_data)
    
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
                
                score_a = cand_a['aggregated_score']
                score_b = cand_b['aggregated_score']
                
                # Skip if scores are too close
                if abs(score_a - score_b) < 0.5:
                    continue
                
                # Randomize order to avoid position bias
                if random.random() < 0.5:
                    cand_a, cand_b = cand_b, cand_a
                    score_a, score_b = score_b, score_a
                
                if score_a > score_b:
                    ground_truth_winner = "Summary A"
                    winner_cand = cand_a
                    loser_cand = cand_b
                else:
                    ground_truth_winner = "Summary B"
                    winner_cand = cand_b
                    loser_cand = cand_a
                
                rationale = create_synthetic_rationale(winner_cand, loser_cand)
                
                user_content = (
                    f"Here is a writing prompt:\n{data['base_prompt']}\n\n"
                    f"Story A:\n{cand_a['summary']}\n\n"
                    f"Story B:\n{cand_b['summary']}\n\n"
                    "Which story is better and why?"
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
    test_size = max(500, int(total_samples * 0.1)) # Ensure at least 500 test samples if possible
    
    if total_samples > test_size:
        test_data = training_examples[:test_size]
        train_data = training_examples[test_size:]
    else:
        print("Warning: Not enough data for split. Using all for train.")
        train_data = training_examples
        test_data = []

    os.makedirs("data", exist_ok=True)
    
    # Save locally
    with open("data/pairwise_train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("data/pairwise_test.jsonl", "w") as f:
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
        
        repo_id = "JasonYan777/creative-writing-pairwise-critic"
        dataset_dict.push_to_hub(repo_id)
        print(f"Successfully pushed to {repo_id}")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")

if __name__ == "__main__":
    construct_comparison_data()
