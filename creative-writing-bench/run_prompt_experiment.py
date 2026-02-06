import os
import sys
import json
import logging
import argparse
import glob
import random
from typing import List, Dict, Any

# Ensure we can import from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from utils.api import APIClient
except ImportError:
    sys.path.append(os.path.join(current_dir, 'creative-writing-bench'))
    try:
        from utils.api import APIClient
    except ImportError:
         print("Could not import APIClient. Please check your path.")
         sys.exit(1)

logging.basicConfig(level=logging.INFO)

# 1. DEFINE EXPERIMENTAL PROMPTS
# These templates must contain {prompt} and {history} placeholders.
EXPERIMENTAL_PROMPTS = {
    # 2. General Improvement
    "make_it_better": (
        "{prompt}\n\n"
        "--- History ---\n"
        "{history}\n"
        "----------------\n"
        "Goal: Make this story significantly better.\n"
        "Instruction: Use the critique in the history as a guide, but your primary goal is simply to write a superior version of this story. "
        "Improve the writing quality, pacing, and impact. Trust your creative instincts.\n"
        "Output ONLY the story."
    ),

    # 3. Fix Weaknesses (Critique Driven)
    "fix_weaknesses": (
        "{prompt}\n\n"
        "HISTORY:\n{history}\n"
        "TASK: Fix the weaknesses identified in the critique.\n"
        "Instruction: Address every point in the feedback history. "
        "Ensure the new draft has no flaws and fully refines the areas that were criticized. "
        "Do not add unnecessary flair; just make it solid and correct.\n"
        "Output the final story text only."
    ),

    # 4. Creative Upgrade (High Quality)
    "creative_upgrade": (
        "{prompt}\n\n"
        "HISTORY:\n{history}\n"
        "TASK: Upgrade the story to a professional standard.\n"
        "Instruction: Take the core concept and execute it with higher artistic quality. "
        "Make the prose elegant, the character voices distinct, and the ending resonant. "
        "Elevate the material.\n"
        "Output the final story text only."
    )
}

def reconstruct_history_text(rounds: List[Dict[str, Any]]) -> str:
    """
    Reconstructs the history text block exactly as api.py does it.
    """
    history_buffer = []
    
    for r_data in rounds:
        round_idx = r_data.get("round", 1)
        challenger = r_data.get("draft_a_challenger", "")
        incumbent = r_data.get("draft_b_incumbent", "")
        evaluation = r_data.get("evaluation", "")
        winning_label = r_data.get("verdict", "Draft B") # Default to Draft B if missing
        
        if round_idx == 1:
            history_entry = (
                f"--- Round 1 ---\n"
                f"Draft A:\n{challenger}\n\n"
                f"Draft B:\n{incumbent}\n\n"
                f"Critic Evaluation:\n{evaluation}\n"
                f"Selected Winner: {winning_label}\n"
            )
        else:
            history_entry = (
                f"--- Round {round_idx} ---\n"
                f"New Challenger (Draft A):\n{challenger}\n\n"
                f"Previous Best (Draft B):\n{incumbent}\n\n"
                f"Critic Evaluation:\n{evaluation}\n"
                f"Selected Winner: {winning_label}\n"
            )
        history_buffer.append(history_entry)
        
    return "\n\n".join(history_buffer)

def load_history_files(base_dir: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Loads random history files from the specified directory.
    """
    files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    if not files:
        print(f"No history files found in {base_dir}")
        return []
        
    print(f"Found {len(files)} history files. Selecting {limit}...")
    selected_files = random.sample(files, min(limit, len(files)))
    
    loaded_data = []
    for fpath in selected_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                # Basic validation
                if "original_prompt" in data and "rounds" in data and "final_response" in data:
                    data["file_path"] = fpath
                    loaded_data.append(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    return loaded_data

def run_experiment():
    gen_client = APIClient(model_type="test")
    judge_client = APIClient(model_type="judge")
    
    model_name = os.getenv("GEMINI_MODEL", "gemini/gemini-3-flash-preview")
    judge_model = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4-5-20250929")
    
    # Locate history directory
    # Based on user context: creative-writing-bench/tinker_history/gemini-3-flash-preview_235b_2__gemini-3-flash-preview
    history_base_dir = os.path.join(current_dir, "tinker_history")
    
    # Load Real Data
    history_items = load_history_files(history_base_dir, limit=20)
    
    if not history_items:
        print("No valid history items loaded. Exiting.")
        return

    results = []
    
    print(f"Starting Offline Refinement Experiment...")
    print(f"Model: {model_name}")

    for item in history_items:
        original_prompt = item["original_prompt"]
        rounds = item["rounds"]
        original_final = item["final_response"]
        item_id = item.get("prompt_id", "unknown")
        
        # Reconstruct History Context
        history_text = reconstruct_history_text(rounds)
        
        print(f"\nProcessing ID: {item_id}")
        
        for prompt_id, prompt_template in EXPERIMENTAL_PROMPTS.items():
            print(f"  Testing prompt: {prompt_id}")
            
            # Construct the FULL prompt that goes to the model
            # This logic mimics what happens inside api.py's lopp when a template is used.
            # But here we are doing a "One Shot Refinement" based on loaded history.
            
            if prompt_template:
                full_prompt = prompt_template.replace("{prompt}", original_prompt).replace("{history}", history_text)
            else:
                 # Default reconstruction logic from api.py
                 full_prompt = (
                    f"{original_prompt}\n\n"
                    f"--- Previous Attempts History ---\n"
                    f"{history_text}\n"
                    f"---------------------------------\n"
                    f"Goal: Surpass the previous best response (Draft B in the most recent round).\n"
                    f"Instruction: Analyze the critique in the history. Keep what was good, but fix the weaknesses identified. "
                    f"Generate a new, improved response that addresses the feedback.\n"
                    f"IMPORTANT constraints:\n"
                    f"- Output ONLY the creative writing piece directly.\n"
                    f"- You MUST make significant improvements. Identical text is a failure. Do not simply reproduce the previous draft.\n"
                    f"- Do NOT include any titles, preambles, analysis, or 'Why this surpasses Draft B' sections.\n"
                    f"- Start directly with the story text."
                )

            try:
                # Generate ONE new draft
                # equivalent to calling generate() but passing our constructed prompt
                output = gen_client.generate(
                    model=model_name,
                    prompt=full_prompt,
                    # We pass system prompt if needed, or other params
                    temperature=0.7 # Using higher temp for variety? Or 0.7 standard?
                )
                
                results.append({
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "new_draft": output,
                    "original_final": original_final,
                    "original_prompt_text": original_prompt
                })
                print(f"    -> Generated {len(output)} chars.")
                
            except Exception as e:
                print(f"    -> Error: {e}")

    # Save Results
    with open("experiment_results_offline.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Judge Results
    judge_results(judge_client, results, judge_model)

def judge_results(api, results, model_name):
    print("\nStarting Judging Phase...")
    
    judge_scores = []
    
    for r in results:
        if "error" in r in r: continue
        
        print(f"  Judging {r['prompt_id']} vs Original Final for {r['item_id']}")
        
        # A = Original Final (Baseline)
        # B = New Draft (Candidate)
        
        prompt = (
            "You are an expert creative writing judge. Compare the following two stories based on the writing prompt.\n\n"
            f"WRITING PROMPT: {r['original_prompt_text']}\n\n"
            f"STORY A (Original Best):\n{r['original_final']}\n\n"
            f"STORY B (New Candidate):\n{r['new_draft']}\n\n"
            "Which story is better? Consider creativity, style, and adherence to the prompt.\n"
            "Reply with ONLY the letter 'A' or 'B' representing the winner, followed by a one-sentence explanation.\n"
            "Format: [Winner]: [Reason]"
        )
        
        try:
            verdict = api.generate(
                model=model_name,
                prompt=prompt,
                max_tokens=150,
                temperature=0.0
            )
            
            print(f"    -> Verdict: {verdict}")
            
            judge_scores.append({
                "item_id": r['item_id'],
                "prompt_id": r['prompt_id'],
                "verdict_raw": verdict,
                "winner": "B" if verdict.strip().upper().startswith("B") or "[WINNER]: B" in verdict.upper() else "A"
            })
        except Exception as e:
            print(f"    -> Judge error: {e}")

    with open("experiment_judgments_offline.json", "w") as f:
        json.dump(judge_scores, f, indent=2)
        
    # Print Ranked Summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS (Ranked by Win Rate)")
    print("="*60)
    
    stats = {}
    for entry in judge_scores:
        pid = entry["prompt_id"]
        verdict = entry.get("verdict_raw", "").strip()
        
        # Robust parsing logic
        winner = "Unknown"
        if entry.get("winner"):
            winner = entry["winner"]
        elif verdict.upper().startswith("A:") or verdict.upper().startswith("A "):
            winner = "A"
        elif verdict.upper().startswith("B:") or verdict.upper().startswith("B "):
            winner = "B"
        elif "[WINNER]: A" in verdict.upper():
            winner = "A"
        elif "[WINNER]: B" in verdict.upper():
            winner = "B"
            
        if pid not in stats:
            stats[pid] = {"total": 0, "wins": 0}
        
        stats[pid]["total"] += 1
        if winner == "B":
            stats[pid]["wins"] += 1
            
    # Calculate rates and sort
    ranked = []
    for pid, s in stats.items():
        rate = (s["wins"] / s["total"]) * 100 if s["total"] > 0 else 0
        ranked.append((pid, s["wins"], s["total"], rate))
        
    # Sort by win rate (descending)
    ranked.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'RANK':<5} | {'PROMPT ID':<25} | {'WINS':<6} | {'TOTAL':<6} | {'WIN RATE':<10}")
    print("-" * 65)
    
    for i, (pid, wins, total, rate) in enumerate(ranked, 1):
        print(f"{i:<5} | {pid:<25} | {wins:<6} | {total:<6} | {rate:.1f}%")
    print("-" * 65)
    print("Note: 'Wins' means the New Candidate (Draft B) beat the Original Best (Draft A).")

if __name__ == "__main__":
    run_experiment()
