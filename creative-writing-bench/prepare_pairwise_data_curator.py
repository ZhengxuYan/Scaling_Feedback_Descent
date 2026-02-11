
import json
import os
import random
import sys
import statistics
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from bespokelabs import curator

# Add parent directory to path to import text_metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creative-writing-bench.text_metrics import compute_metrics

load_dotenv()

# Metrics where lower is better
NEGATIVE_METRICS = [
    'Meandering', 'Weak Dialogue', "Tell-Don't-Show", 'Unsurprising or Uncreative',
    'Amateurish', 'Purple Prose', 'Overwrought', 'Incongruent Ending Positivity',
    'Unearned Transformations',
    'Slop Score'
]

class CreativeWritingJudge(curator.LLM):
    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        prompt_text = input["prompt"]
        draft_a = input["draft_a"]
        draft_b = input["draft_b"]
        scores_a = input["scores_a"]
        scores_b = input["scores_b"]
        
        # Format scores for display
        def format_scores(scores):
            lines = []
            for k, v in scores.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.2f}")
                else:
                    lines.append(f"- {k}: {v}")
            return "\n".join(lines)

        scores_text_a = format_scores(scores_a)
        scores_text_b = format_scores(scores_b)
        
        user_content = (
            f"Here is a creative writing prompt:\n{prompt_text}\n\n"
            f"Draft A:\n{draft_a}\n\n"
            f"Draft B:\n{draft_b}\n\n"
            f"Here are the detailed metric scores for both drafts as a reference:\n\n"
            f"Draft A Scores:\n{scores_text_a}\n\n"
            f"Draft B Scores:\n{scores_text_b}\n\n"
            "Please evaluate these two drafts. Which one is better? Provide a verdict and a clear, natural language rationale explaining your decision. "
            "You can use the provided scores as a reference to inform your judgment, but ensure your rationale is grounded in the text itself. "
            "You have the autonomy to make your own decision based on the text quality.\n"
            "Output valid JSON with keys 'verdict' (either 'Draft A' or 'Draft B') and 'rationale'."
        )
        
        return [
            {"role": "system", "content": "You are an expert creative writing critic. Output JSON only."},
            {"role": "user", "content": user_content}
        ]

    def parse(self, input: Dict, response: str) -> Dict:
        try:
            # Clean response to ensure it's valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            # Allow control characters (newlines) in strings
            data = json.loads(response, strict=False)
            model_verdict = data.get("verdict", "Unknown")
            model_rationale = data.get("rationale", "")
            
        except Exception as e:
            model_verdict = "Unknown"
            model_rationale = ""
            print(f"Error parsing response: {e}")
            
        ground_truth_winner = input["ground_truth_winner"]
        
        # Normalize verdict string
        if "draft a" in model_verdict.lower():
            clean_verdict = "Draft A"
        elif "draft b" in model_verdict.lower():
            clean_verdict = "Draft B"
        else:
            clean_verdict = "Unknown"
            
        is_agreement = (clean_verdict == ground_truth_winner)
        
        return {
            **input,
            "model_verdict": clean_verdict,
            "model_rationale": model_rationale,
            "is_agreement": is_agreement
        }

def load_run_data(filepath: str) -> Dict[str, Any]:
    print(f"Loading run data from {filepath}...")
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_aggregated_score(scores: Dict[str, float]) -> float:
    total_score = 0.0
    count = 0
    
    for metric, val in scores.items():
        if metric == "Overall Impression": 
            continue
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
    grouped_data = {}
    
    print("Extracting candidates...")
    for run_key, run_info in run_data.items():
        test_model = run_info.get("test_model", "unknown")
        creative_tasks = run_info.get("creative_tasks", {})
        
        for iter_key, prompts_dict in creative_tasks.items():
            for prompt_id, task_info in prompts_dict.items():
                
                if task_info.get("status") not in ["completed", "judged"]:
                    continue
                
                base_prompt = task_info.get("base_prompt", "")
                results_by_mod = task_info.get("results_by_modifier", {})
                
                for mod_key, res_block in results_by_mod.items():
                    judge_scores = res_block.get("judge_scores", {})
                    model_response = res_block.get("model_response", "")
                    
                    if not judge_scores or not model_response:
                        continue
                        
                    overall_score = judge_scores.get("Overall Impression", 0)
                    aggregated_score = calculate_aggregated_score(judge_scores)
                    
                    try:
                        metrics_res = compute_metrics(model_response, metrics=["slop_score"])
                        slop_score = metrics_res["slop_score"]
                    except Exception:
                        slop_score = 0.0
                    
                    if prompt_id not in grouped_data:
                        grouped_data[prompt_id] = {
                            "base_prompt": base_prompt,
                            "candidates": []
                        }
                    
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
    print("Normalizing scores...")
    
    all_agg_scores = []
    all_slop_scores = []
    
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
    
    for prompt_id, data in grouped_data.items():
        for cand in data['candidates']:
            raw_agg = cand['aggregated_score']
            raw_slop = cand['slop_score']
            
            z_agg = (raw_agg - mean_agg) / std_agg
            z_slop = (raw_slop - mean_slop) / std_slop
            
            final_score = z_agg - z_slop
            cand['final_score'] = final_score

def create_synthetic_rationale(winner_cand, loser_cand):
    w_scores = winner_cand['scores']
    l_scores = loser_cand['scores']
    
    details = []
    positive_metrics = [
        'Adherence to Instructions', 'Believable Character Actions', 'Nuanced Characters',
        'Consistent Voice/Tone of Writing', 'Imagery and Descriptive Quality', 'Elegant Prose',
        'Emotionally Engaging', 'Emotionally Complex', 'Coherent', 'Well-earned Lightness or Darkness',
        'Sentences Flow Naturally', 'Overall Reader Engagement'
    ]
    
    diffs = []
    for m in positive_metrics:
        w_val = float(w_scores.get(m, 0))
        l_val = float(l_scores.get(m, 0))
        diff = w_val - l_val
        if diff != 0:
            if diff > 0: diffs.append((diff, f"better {m}", w_val, l_val))
            else: diffs.append((diff, f"worse {m}", w_val, l_val))
            
    for m in NEGATIVE_METRICS:
        w_val = float(w_scores.get(m, 0))
        l_val = float(l_scores.get(m, 0))
        diff = l_val - w_val 
        if diff != 0:
            if diff > 0: diffs.append((diff, f"less {m}", w_val, l_val))
            else: diffs.append((diff, f"more {m}", w_val, l_val))
            
    diffs.sort(key=lambda x: x[0], reverse=True)
    
    for _, reason, w_val, l_val in diffs:
        details.append(f"{reason} ({w_val:.2f} vs {l_val:.2f})")
            
    reason_text = ", ".join(details) if details else "better overall quality"
    return f"The winner produced a better draft because it has {reason_text}. (Final Score: {winner_cand['final_score']:.2f} vs {loser_cand['final_score']:.2f})"

def construct_comparison_data():
    runs_file = "creative_bench_runs.json"
    if not os.path.exists(runs_file):
        print(f"Error: {runs_file} not found.")
        return

    run_data = load_run_data(runs_file)
    grouped_data = extract_candidates(run_data)
    normalize_scores_and_compute_final(grouped_data)
    
    # Initialize Judge
    judge = CreativeWritingJudge(model_name="gpt-5-mini")
    
    # Intermediate file for incremental saving
    intermediate_file = "verified_pairs.jsonl"
    
    # Clear intermediate file if it exists to start fresh
    with open(intermediate_file, "w") as f:
        pass

    total_pairs_count = 0
    total_agreement_count = 0
    
    print("Starting batch processing per prompt...")
    
    prompts_list = list(grouped_data.items())
    total_prompts = len(prompts_list)
    
    for idx, (prompt_id, data) in enumerate(prompts_list):
        cands = data['candidates']
        if len(cands) < 2:
            continue
            
        # Generate pairs for this specific prompt
        prompt_pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                cand_a = cands[i]
                cand_b = cands[j]
                
                score_a = cand_a.get('final_score', 0)
                score_b = cand_b.get('final_score', 0)
                
                if abs(score_a - score_b) < 0.5:
                    continue
                
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
                
                prompt_pairs.append({
                    "prompt": data['base_prompt'],
                    "draft_a": cand_a['summary'],
                    "draft_b": cand_b['summary'],
                    "scores_a": cand_a['scores'],
                    "scores_b": cand_b['scores'],
                    "ground_truth_winner": ground_truth_winner,
                    "winner_cand": winner_cand,
                    "loser_cand": loser_cand
                })
        
        # Downsample if needed (Restore 500 limit per prompt)
        if len(prompt_pairs) > 50:
             prompt_pairs = random.sample(prompt_pairs, 50)
             
        if not prompt_pairs:
            continue
            
        print(f"[{idx+1}/{total_prompts}] Processing prompt {prompt_id} with {len(prompt_pairs)} pairs...")
        
        # Run Curator for this batch
        try:
            results = judge(prompt_pairs)
            
            if hasattr(results, "dataset"):
                results_list = list(results.dataset)
            else:
                results_list = list(results)
                
            batch_agreement = 0
            
            with open(intermediate_file, "a") as f:
                for res in results_list:
                    if res["is_agreement"]:
                        batch_agreement += 1
                        
                        user_content = (
                            f"Here is a writing prompt:\n{res['prompt']}\n\n"
                            f"Draft A:\n{res['draft_a']}\n\n"
                            f"Draft B:\n{res['draft_b']}\n\n"
                            "Which draft is better and why?"
                        )
                        
                        assistant_content = f"**Verdict**: {res['ground_truth_winner']}\n\n**Feedback**: {res['model_rationale']}"
                        
                        entry = {
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content}
                            ]
                        }
                        f.write(json.dumps(entry) + "\n")
            
            total_pairs_count += len(prompt_pairs)
            total_agreement_count += batch_agreement
            print(f"  Batch Agreement: {batch_agreement}/{len(prompt_pairs)}")
            
        except Exception as e:
            print(f"  Error processing prompt {prompt_id}: {e}")
            
    print("--------------------------------------------------")
    if total_pairs_count > 0:
        print(f"Total Agreement rate: {total_agreement_count}/{total_pairs_count} ({total_agreement_count/total_pairs_count*100:.2f}%)")
    else:
        print("No pairs processed.")

    print(f"Loading verified pairs from {intermediate_file}...")
    
    # Load all verified pairs for splitting
    training_examples = []
    if os.path.exists(intermediate_file):
        with open(intermediate_file, "r") as f:
            for line in f:
                if line.strip():
                    training_examples.append(json.loads(line))
    
    print(f"Total verified comparison pairs: {len(training_examples)}")
    
    if not training_examples:
        print("No data generated. Exiting.")
        return

    # Shuffle
    random.shuffle(training_examples)

    # Split into train and test
    total_samples = len(training_examples)
    test_size = max(50, int(total_samples * 0.1)) # Ensure reasonable test size
    
    if total_samples > test_size:
        test_data = training_examples[:test_size]
        train_data = training_examples[test_size:]
    else:
        print("Warning: Not enough data for split. Using all for train.")
        train_data = training_examples
        test_data = []

    os.makedirs("data", exist_ok=True)
    
    # Save final splits locally
    with open("data/pairwise_train_freeform_sub.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("data/pairwise_test_freeform_sub.jsonl", "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")

    print("Data preparation complete.")
    
    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    try:
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data)
        })
        
        repo_id = "JasonYan777/creative-writing-pairwise-critic-freeform_sub"
        dataset_dict.push_to_hub(repo_id)
        print(f"Successfully pushed to {repo_id}")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")

if __name__ == "__main__":
    construct_comparison_data()
