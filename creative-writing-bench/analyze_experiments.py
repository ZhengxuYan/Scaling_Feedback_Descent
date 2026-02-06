
import json
import os
import re

def analyze_judgments():
    file_path = "experiment_judgments_offline.json"
    if not os.path.exists(file_path):
        print("No judgment file found.")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    stats = {}

    print(f"\nAnalying {len(data)} judgment records...")

    for entry in data:
        pid = entry["prompt_id"]
        verdict = entry.get("verdict_raw", "").strip()
        
        # Robust Parsing Logic
        winner = "Unknown"
        
        # Regex to catch "A" or "B" at start, optionally followed by colon or space
        # Or look for [Winner]: X format inside text
        
        match_start = re.match(r"^(A|B)[:\s]", verdict, re.IGNORECASE)
        match_bracket = re.search(r"\[Winner\]:?\s*(A|B)", verdict, re.IGNORECASE)
        
        if match_start:
            winner = match_start.group(1).upper()
        elif match_bracket:
            winner = match_bracket.group(1).upper()
        elif verdict.upper().startswith("A"):
             winner = "A" # Fallback for just "A"
        elif verdict.upper().startswith("B"):
             winner = "B" # Fallback for just "B"
        
        # Initialize stats for this prompt ID if needed
        if pid not in stats:
            stats[pid] = {"total": 0, "wins": 0, "losses": 0, "ties": 0}
            
        if winner == "B":
            stats[pid]["wins"] += 1
            stats[pid]["total"] += 1
        elif winner == "A":
            stats[pid]["losses"] += 1
            stats[pid]["total"] += 1
        else:
            stats[pid]["ties"] += 1
            stats[pid]["total"] += 1 # Counting unknown/ties in total? Yes.
            # print(f"Warning: Could not determine winner for: {verdict[:50]}...")

    # Sort and Print
    print("\n" + "="*65)
    print("CORRECTED EXPERIMENT RESULTS (Ranked by Win Rate)")
    print("="*65)
    print(f"{'RANK':<5} | {'PROMPT ID':<25} | {'WINS':<6} | {'ALL':<5} | {'WIN RATE':<10}")
    print("-" * 65)

    ranked = []
    for pid, s in stats.items():
        rate = (s["wins"] / s["total"]) * 100 if s["total"] > 0 else 0
        ranked.append((pid, s["wins"], s["total"], rate))
        
    ranked.sort(key=lambda x: x[3], reverse=True)
    
    for i, (pid, wins, total, rate) in enumerate(ranked, 1):
        print(f"{i:<5} | {pid:<25} | {wins:<6} | {total:<5} | {rate:.1f}%")
    print("-" * 65)
    print("Note: 'Wins' means the New Candidate (Draft B) beat the Original Best (Draft A).")

if __name__ == "__main__":
    analyze_judgments()
