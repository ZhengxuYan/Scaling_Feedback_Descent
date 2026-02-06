
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
from text_metrics import compute_metrics

load_dotenv()

# Metrics where lower is better
NEGATIVE_METRICS = [
    'Meandering', 'Weak Dialogue', "Tell-Don't-Show", 'Unsurprising or Uncreative',
    'Amateurish', 'Purple Prose', 'Overwrought', 'Incongruent Ending Positivity',
    'Unearned Transformations',
    'Slop Score'
]

class RationaleGenerator(curator.LLM):
    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        prompt_text = input["prompt"]
        draft_a = input["draft_a"]
        draft_b = input["draft_b"]
        scores_a = input["scores_a"]
        scores_b = input["scores_b"]
        salt = input.get("salt", "")
        
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
        
        system_content = "You are an expert creative writing critic. Output JSON only."
        if salt:
            system_content += f" (Seed: {salt})"

        return [
            {"role": "system", "content": system_content},
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
            # print(f"Error parsing response: {e}")
            
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

class DraftImprover(curator.LLM):
    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        prompt_text = input["prompt"]
        draft_a = input["draft_a"]
        draft_b = input["draft_b"]
        rationale = input["rationale_to_use"]
        
        user_content = (
            f"Here is a creative writing prompt:\n{prompt_text}\n\n"
            f"Draft A:\n{draft_a}\n\n"
            f"Draft B:\n{draft_b}\n\n"
            f"A critic has evaluated these drafts and provided the following rationale for why one is better:\n"
            f"\"{rationale}\"\n\n"
            f"Task: Write a NEW, IMPROVED draft that surpasses both Draft A and Draft B. "
            f"Analyze the rationale to understand the strengths of the winner and the weaknesses of the loser. "
            f"Incorporate the strengths and fix the weaknesses to create a superior piece of writing.\n"
            f"Output ONLY the new draft text."
        )
        
        return [
            {"role": "system", "content": "You are an expert creative writer."},
            {"role": "user", "content": user_content}
        ]

    def parse(self, input: Dict, response: str) -> Dict:
        # Just return the raw text response as the improved draft
        return {
            **input,
            "improved_draft": response.strip()
        }

class ImprovementJudge(curator.LLM):
    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        prompt_text = input["prompt"]
        improved_1 = input["improved_draft_1"]
        improved_2 = input["improved_draft_2"]
        
        user_content = (
            f"Here is a creative writing prompt:\n{prompt_text}\n\n"
            f"Improved Draft 1:\n{improved_1}\n\n"
            f"Improved Draft 2:\n{improved_2}\n\n"
            f"Two writers tried to improve upon previous drafts based on different critiques. "
            f"Which of these two NEW drafts is better? "
            f"Output valid JSON with key 'winner' (either '1' or '2') and 'reason'."
        )
        
        return [
            {"role": "system", "content": "You are an expert creative writing critic. Output JSON only."},
            {"role": "user", "content": user_content}
        ]

    def parse(self, input: Dict, response: str) -> Dict:
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            data = json.loads(response, strict=False)
            winner = str(data.get("winner", "0"))
            
            if "1" in winner:
                clean_winner = "1"
            elif "2" in winner:
                clean_winner = "2"
            else:
                clean_winner = "0"
                
        except Exception:
            clean_winner = "0"
            
        return {
            **input,
            "improvement_winner": clean_winner
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

def construct_comparison_data():
    runs_file = "creative-writing-bench/creative_bench_runs.json"
    if not os.path.exists(runs_file):
        runs_file = "creative_bench_runs.json"
        
    if not os.path.exists(runs_file):
        print(f"Error: {runs_file} not found.")
        return

    run_data = load_run_data(runs_file)
    grouped_data = extract_candidates(run_data)
    normalize_scores_and_compute_final(grouped_data)
    
    # Initialize Curator Models
    rationale_gen = RationaleGenerator(model_name="gpt-5-mini") 
    draft_improver = DraftImprover(model_name="gpt-5-mini")
    improvement_judge = ImprovementJudge(model_name="gpt-5-mini")
    
    intermediate_file = "verified_pairs_improvement.jsonl"
    
    # Check resumption
    processed_count = 0
    if os.path.exists(intermediate_file):
        with open(intermediate_file, "r") as f:
            processed_count = sum(1 for line in f)
    print(f"Found {processed_count} existing entries in {intermediate_file}. Appending new ones...")

    total_pairs_count = 0
    total_added_count = 0
    
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
                else:
                    ground_truth_winner = "Draft B"
                
                prompt_pairs.append({
                    "prompt": data['base_prompt'],
                    "draft_a": cand_a['summary'],
                    "draft_b": cand_b['summary'],
                    "scores_a": cand_a['scores'],
                    "scores_b": cand_b['scores'],
                    "ground_truth_winner": ground_truth_winner,
                })
        
        # Limit processing
        # Downsample if needed (Restore 100 limit per prompt)
        if len(prompt_pairs) > 50:
             prompt_pairs = random.sample(prompt_pairs, 50)
             
        if not prompt_pairs:
            continue
            
        print(f"[{idx+1}/{total_prompts}] Processing prompt {prompt_id} with {len(prompt_pairs)} pairs...")
        
        try:
            # --- STAGE 1: Generate Rationale 1 ---
            # Add salt to inputs to ensure distinct caching/generation
            prompt_pairs_1 = [dict(p, salt="A") for p in prompt_pairs]
            results_1 = rationale_gen(prompt_pairs_1)
            if hasattr(results_1, "dataset"): results_1 = list(results_1.dataset)
            else: results_1 = list(results_1)
            
            # --- STAGE 2: Generate Rationale 2 ---
            prompt_pairs_2 = [dict(p, salt="B") for p in prompt_pairs]
            results_2 = rationale_gen(prompt_pairs_2)
            if hasattr(results_2, "dataset"): results_2 = list(results_2.dataset)
            else: results_2 = list(results_2)
            
            # --- STAGE 3: Filter & Improve ---
            
            # Prepare data for improvement
            # We only keep pairs where BOTH rationales agreed with ground truth
            improvement_inputs_1 = []
            improvement_inputs_2 = []
            valid_indices = []
            
            # Ensure results align with original prompt_pairs
            # Curator usually preserves order. We assume 1-to-1 mapping.
            
            for k in range(len(prompt_pairs)):
                r1 = results_1[k]
                r2 = results_2[k]
                
                if r1["is_agreement"] and r2["is_agreement"]: # Both must be correct
                    # Check if rationales are identical (rare but possible with cache/luck)
                    if r1["model_rationale"] == r2["model_rationale"]:
                        # print("    Duplicate rationale filtered.")
                        continue

                    valid_indices.append(k)
                    
                    base_input = prompt_pairs[k] # Contains prompt, draft_a, draft_b...
                    
                    inp1 = base_input.copy()
                    inp1["rationale_to_use"] = r1["model_rationale"]
                    improvement_inputs_1.append(inp1)
                    
                    inp2 = base_input.copy()
                    inp2["rationale_to_use"] = r2["model_rationale"]
                    improvement_inputs_2.append(inp2)
            
            if not valid_indices:
                print("  No pairs passed dual-agreement filter.")
                continue
                
            print(f"  {len(valid_indices)} pairs passed filter. Generating improvements...")
            
            # Generate Improvements (Batch)
            improved_drafts_1 = draft_improver(improvement_inputs_1)
            improved_drafts_2 = draft_improver(improvement_inputs_2)
            
            if hasattr(improved_drafts_1, "dataset"): list_imp_1 = list(improved_drafts_1.dataset)
            else: list_imp_1 = list(improved_drafts_1)
            
            if hasattr(improved_drafts_2, "dataset"): list_imp_2 = list(improved_drafts_2.dataset)
            else: list_imp_2 = list(improved_drafts_2)
            
            # --- STAGE 4: Compare Improvements ---
            judge_inputs = []
            for k in range(len(valid_indices)):
                # Original pair data
                orig_idx = valid_indices[k]
                orig_pair = prompt_pairs[orig_idx]
                
                # Improved drafts
                id1 = list_imp_1[k]["improved_draft"]
                id2 = list_imp_2[k]["improved_draft"]
                
                j_inp = {
                    "prompt": orig_pair["prompt"],
                    "improved_draft_1": id1,
                    "improved_draft_2": id2,
                    "original_pair_data": orig_pair,
                    "rationale_1": results_1[orig_idx]["model_rationale"],
                    "rationale_2": results_2[orig_idx]["model_rationale"]
                }
                judge_inputs.append(j_inp)
                
            print(f"  Judging {len(judge_inputs)} improvement pairs...")
            judgement_results = improvement_judge(judge_inputs)
            
            if hasattr(judgement_results, "dataset"): final_judgments = list(judgement_results.dataset)
            else: final_judgments = list(judgement_results)
            
            # --- Final Save ---
            batch_added = 0
            with open(intermediate_file, "a") as f:
                for res in final_judgments:
                    winner = res["improvement_winner"]
                    
                    final_rationale = None
                    if winner == "1":
                        final_rationale = res["rationale_1"]
                    elif winner == "2":
                        final_rationale = res["rationale_2"]
                    
                    if final_rationale:
                        # Construct final training entry
                        orig = res["original_pair_data"]
                        
                        user_content = (
                            f"Here is a writing prompt:\n{orig['prompt']}\n\n"
                            f"Draft A:\n{orig['draft_a']}\n\n"
                            f"Draft B:\n{orig['draft_b']}\n\n"
                            "Which draft is better and why?"
                        )
                        
                        assistant_content = f"**Verdict**: {orig['ground_truth_winner']}\n\n**Feedback**: {final_rationale}"
                        
                        entry = {
                            "messages": [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": assistant_content}
                            ]
                        }
                        f.write(json.dumps(entry) + "\n")
                        batch_added += 1
            
            total_added_count += batch_added
            print(f"  Added {batch_added} verified pairs with high-quality rationales.")
            
        except Exception as e:
            print(f"  Error processing prompt {prompt_id}: {e}")
            import traceback
            traceback.print_exc()

    print("--------------------------------------------------")
    print(f"Total pairs added: {total_added_count}")

    # (Post-processing split/save logic similar to original can be added here if needed)
    # For now, we just save to intermediate file as requested.

if __name__ == "__main__":
    construct_comparison_data()
